"""
Micro-Transformer Graph Reasoner — Phase 5
============================================
A tiny transformer (~50K params) that reasons over the knowledge graph.
Takes (question + graph triples) as input, outputs an answer.

Three answer modes:
  1. Entity pointer — "Who is my wife?" → Prabhashi
  2. Boolean — "Do I have a brother?" → No
  3. Count — "How many sisters?" → 2

Training data is 100% synthetic — generated from graph triples with
paraphrase augmentation. ~200-500 QA pairs from a small personal graph.

Architecture:
  Token embedding (64-dim) → Transformer encoder (2 layers, 4 heads)
  → Answer mode classifier → Mode-specific head
"""

import os
import re
import json
import math
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from limbiq.graph.store import GraphStore, Entity, Relation

logger = logging.getLogger(__name__)


# ─── Dynamic Vocabulary ──────────────────────────────────────────

# Fixed question-word tokens (always present)
QUESTION_WORDS = [
    "[PAD]", "[CLS]", "[SEP]", "[T]", "[UNK]",
    "[YES]", "[NO]",
    "[COUNT_0]", "[COUNT_1]", "[COUNT_2]", "[COUNT_3]", "[COUNT_4]",
    "[COUNT_5]", "[COUNT_6]", "[COUNT_7]", "[COUNT_8]", "[COUNT_9]", "[COUNT_10]",
    # Question words
    "who", "what", "where", "how", "is", "are", "my", "do", "does",
    "have", "many", "tell", "me", "about", "the", "a", "i",
    "'s", "name", "work", "live", "related", "to", "of", "at", "in",
    "whose", "which", "am", "married", "called", "for",
    "yes", "no", "any", "does", "did", "can",
]


class GraphVocab:
    """
    Dynamic vocabulary built from graph state.
    Fixed question words + entity names + relation predicates.
    Rebuilds automatically when graph changes.
    """

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.n_entities = 0
        self._entity_id_to_pointer = {}   # graph entity ID → pointer (0..n-1)
        self._pointer_to_entity_name = {} # pointer → entity name
        self._entity_id_to_vocab = {}     # graph entity ID → vocab index

    def build(self, graph: GraphStore):
        """Build vocabulary from current graph state."""
        self.token_to_idx = {}
        self.idx_to_token = {}
        self._entity_id_to_pointer = {}
        self._pointer_to_entity_name = {}
        self._entity_id_to_vocab = {}

        # Add fixed tokens
        for token in QUESTION_WORDS:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

        # Add relation predicates
        relations = graph.get_all_relations(include_inferred=True)
        predicates = sorted(set(r.predicate for r in relations))
        for pred in predicates:
            token = pred.lower().replace(" ", "_")
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
            # Also add without underscores for matching
            for part in token.split("_"):
                if part and part not in self.token_to_idx:
                    idx = len(self.token_to_idx)
                    self.token_to_idx[part] = idx
                    self.idx_to_token[idx] = part

        # Add entity names to vocab + build separate pointer mapping
        entities = graph.get_all_entities()
        pointer_idx = 0

        for entity in sorted(entities, key=lambda e: e.name.lower()):
            token = entity.name.lower()
            # Add to vocab if not already present
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
            self._entity_id_to_vocab[entity.id] = self.token_to_idx[token]
            # Also add individual words from multi-word names
            for part in token.split():
                if part and part not in self.token_to_idx:
                    idx = len(self.token_to_idx)
                    self.token_to_idx[part] = idx
                    self.idx_to_token[idx] = part

            # Pointer is a simple sequential index, independent of vocab position
            self._entity_id_to_pointer[entity.id] = pointer_idx
            self._pointer_to_entity_name[pointer_idx] = entity.name
            pointer_idx += 1

        self.n_entities = len(entities)

        return self

    def tokenize(self, text: str) -> list[int]:
        """Convert text to token indices."""
        # Simple whitespace + punctuation tokenizer
        text = text.lower().strip()
        # Split possessives
        text = text.replace("'s", " 's")
        text = text.replace("'s", " 's")
        # Split on whitespace and punctuation
        tokens = re.findall(r"[a-z_]+(?:_[a-z]+)*|'s|\d+", text)
        unk = self.token_to_idx.get("[UNK]", 4)
        return [self.token_to_idx.get(t, unk) for t in tokens]

    def encode_triple(self, subj_name: str, pred: str, obj_name: str) -> list[int]:
        """Encode a graph triple as token indices: [T] subj pred obj"""
        t_idx = self.token_to_idx["[T]"]
        unk = self.token_to_idx.get("[UNK]", 4)

        subj_tokens = [self.token_to_idx.get(w, unk) for w in subj_name.lower().split()]
        pred_tokens = [self.token_to_idx.get(w, unk) for w in pred.lower().replace("_", " ").split()]
        obj_tokens = [self.token_to_idx.get(w, unk) for w in obj_name.lower().split()]

        return [t_idx] + subj_tokens + pred_tokens + obj_tokens

    def entity_id_to_pointer(self, entity_id: str) -> int:
        """Convert a graph entity ID to a pointer index (0-based among entities)."""
        return self._entity_id_to_pointer.get(entity_id, -1)

    def pointer_to_entity_name(self, pointer: int) -> str:
        """Convert a pointer index back to entity name."""
        return self._pointer_to_entity_name.get(pointer, "[UNK]")

    @property
    def size(self) -> int:
        return len(self.token_to_idx)


# ─── Transformer Model ──────────────────────────────────────────

class GraphTransformer(nn.Module):
    """
    Micro-transformer for graph QA.

    Input: [CLS] question tokens [SEP] [T] s p o [T] s p o ...
    Output: answer via one of three heads (entity/bool/count)
    """

    def __init__(self, vocab_size: int, n_entities: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_seq_len: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_entities = n_entities

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Answer mode classifier: entity(0), boolean(1), count(2)
        self.mode_head = nn.Linear(d_model, 3)

        # Mode-specific heads
        self.entity_head = nn.Linear(d_model, n_entities)  # pointer over entities
        self.boolean_head = nn.Linear(d_model, 1)          # sigmoid → yes/no
        self.count_head = nn.Linear(d_model, 6)            # classify 0-5

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) 1=real, 0=pad

        Returns dict with:
            mode_logits: (batch, 3)
            entity_logits: (batch, n_entities)
            boolean_logits: (batch, 1)
            count_logits: (batch, 6) — classification 0-5
            cls_hidden: (batch, d_model) — CLS token representation
        """
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Create causal mask for padding
        if attention_mask is not None:
            # Convert 0/1 mask to transformer format (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Use [CLS] token (position 0) for classification
        cls_hidden = hidden[:, 0, :]

        return {
            "mode_logits": self.mode_head(cls_hidden),
            "entity_logits": self.entity_head(cls_hidden),
            "boolean_logits": self.boolean_head(cls_hidden),
            "count_logits": self.count_head(cls_hidden),
            "cls_hidden": cls_hidden,
        }


# ─── Synthetic QA Generator ─────────────────────────────────────

@dataclass
class QASample:
    """A single training/eval sample."""
    question: str
    answer_mode: int      # 0=entity, 1=boolean, 2=count
    entity_answer: int    # pointer index (-1 if not entity mode)
    boolean_answer: int   # 0=no, 1=yes (-1 if not bool mode)
    count_answer: int     # 0-10 (-1 if not count mode)
    hops: int             # number of reasoning hops required
    source_triples: list  # which triples are needed


class SyntheticQAGenerator:
    """
    Generate training data from graph triples.
    No LLM needed — pure template-based generation with paraphrase augmentation.
    """

    # Paraphrase templates per query type
    SINGLE_HOP_TEMPLATES = [
        "who is my {rel}",
        "what is my {rel}'s name",
        "tell me who my {rel} is",
        "my {rel} is who",
        "do you know my {rel}",
        "what's my {rel}'s name",
        "who's my {rel}",
    ]

    WORK_TEMPLATES = [
        "where do i work",
        "what company do i work at",
        "where am i working",
        "tell me where i work",
        "what is my workplace",
        "which company do i work for",
    ]

    ROLE_TEMPLATES = [
        "what do i do",
        "what is my role",
        "what do i do for work",
        "what is my job",
        "what's my profession",
        "what am i",
    ]

    BOOLEAN_HAS_TEMPLATES = [
        "do i have a {rel}",
        "have i got a {rel}",
        "is there a {rel} for me",
        "do i have any {rel}",
    ]

    BOOLEAN_IS_TEMPLATES = [
        "is {name} my {rel}",
        "is my {rel} {name}",
        "am i married to {name}",  # only for wife/husband
    ]

    COUNT_TEMPLATES = [
        "how many {rel_plural} do i have",
        "how many {rel_plural} have i got",
        "what is the number of my {rel_plural}",
        "count my {rel_plural}",
        "tell me how many {rel_plural} i have",
        "do i have any {rel_plural} and how many",
    ]

    MULTI_HOP_TEMPLATES = [
        "who is my {rel1}'s {rel2}",
        "what is my {rel1}'s {rel2}'s name",
        "tell me my {rel1}'s {rel2}",
    ]

    INVERSE_TEMPLATES = [
        "whose {rel} is {name}",
        "who is {name} the {rel} of",
        "{name} is whose {rel}",
    ]

    RELATION_PLURALS = {
        "sister": "sisters",
        "brother": "brothers",
        "son": "sons",
        "daughter": "daughters",
        "wife": "wives",
        "husband": "husbands",
        "friend": "friends",
        "colleague": "colleagues",
    }

    FAMILY_RELS = {"father", "mother", "wife", "husband", "sister", "brother",
                   "son", "daughter"}

    def __init__(self, graph: GraphStore, vocab: GraphVocab, user_name: str = "Dimuthu"):
        self.graph = graph
        self.vocab = vocab
        self.user_name = user_name

    def generate_all(self) -> list[QASample]:
        """Generate all QA pairs from current graph state."""
        samples = []

        entities = self.graph.get_all_entities()
        eid = {e.id: e for e in entities}
        all_rels = self.graph.get_all_relations(include_inferred=True)

        user = self.graph.find_entity_by_name(self.user_name)
        if not user:
            return samples

        # User's outgoing relations
        user_rels = [r for r in all_rels if r.subject_id == user.id]

        # 1. Single-hop entity queries
        samples.extend(self._gen_single_hop(user, user_rels, eid))

        # 2. Boolean existence queries
        samples.extend(self._gen_boolean(user, user_rels, eid))

        # 3. Count queries
        samples.extend(self._gen_count(user, user_rels, eid))

        # 4. Multi-hop queries
        samples.extend(self._gen_multi_hop(user, user_rels, all_rels, eid))

        # 5. Inverse queries
        samples.extend(self._gen_inverse(user, all_rels, eid))

        random.shuffle(samples)
        return samples

    def _gen_single_hop(self, user: Entity, user_rels: list,
                        eid: dict) -> list[QASample]:
        samples = []
        for rel in user_rels:
            obj = eid.get(rel.object_id)
            if not obj:
                continue
            pointer = self.vocab.entity_id_to_pointer(obj.id)
            if pointer < 0:
                continue

            if rel.predicate == "works_at":
                templates = self.WORK_TEMPLATES
            elif rel.predicate == "role":
                templates = self.ROLE_TEMPLATES
            elif rel.predicate in self.FAMILY_RELS:
                templates = self.SINGLE_HOP_TEMPLATES
            else:
                templates = [f"what is my {rel.predicate}"]

            for tmpl in templates:
                q = tmpl.format(rel=rel.predicate)
                samples.append(QASample(
                    question=q,
                    answer_mode=0,
                    entity_answer=pointer,
                    boolean_answer=-1,
                    count_answer=-1,
                    hops=1,
                    source_triples=[(user.name, rel.predicate, obj.name)],
                ))
        return samples

    def _gen_boolean(self, user: Entity, user_rels: list,
                     eid: dict) -> list[QASample]:
        samples = []
        present_preds = set(r.predicate for r in user_rels)

        # Positive: relations that exist
        for rel in user_rels:
            obj = eid.get(rel.object_id)
            if not obj or rel.predicate not in self.FAMILY_RELS:
                continue
            for tmpl in self.BOOLEAN_HAS_TEMPLATES:
                samples.append(QASample(
                    question=tmpl.format(rel=rel.predicate),
                    answer_mode=1,
                    entity_answer=-1,
                    boolean_answer=1,
                    count_answer=-1,
                    hops=1,
                    source_triples=[],
                ))
            # "Is X my Y?" — correct
            for tmpl in self.BOOLEAN_IS_TEMPLATES[:2]:  # skip "am i married to" for non-spouse
                samples.append(QASample(
                    question=tmpl.format(name=obj.name, rel=rel.predicate),
                    answer_mode=1,
                    entity_answer=-1,
                    boolean_answer=1,
                    count_answer=-1,
                    hops=1,
                    source_triples=[(user.name, rel.predicate, obj.name)],
                ))

        # Negative: relations that don't exist
        missing = self.FAMILY_RELS - present_preds
        for rel_type in missing:
            for tmpl in self.BOOLEAN_HAS_TEMPLATES:
                samples.append(QASample(
                    question=tmpl.format(rel=rel_type),
                    answer_mode=1,
                    entity_answer=-1,
                    boolean_answer=0,
                    count_answer=-1,
                    hops=1,
                    source_triples=[],
                ))

        # "Is X my Y?" — wrong entity
        for rel in user_rels:
            obj = eid.get(rel.object_id)
            if not obj or rel.predicate not in self.FAMILY_RELS:
                continue
            # Pick a wrong entity
            wrong_entities = [e for e in eid.values()
                              if e.id != obj.id and e.id != user.id
                              and e.entity_type == "person"]
            for wrong in wrong_entities[:2]:
                samples.append(QASample(
                    question=f"is {wrong.name} my {rel.predicate}",
                    answer_mode=1,
                    entity_answer=-1,
                    boolean_answer=0,
                    count_answer=-1,
                    hops=1,
                    source_triples=[],
                ))

        return samples

    def _gen_count(self, user: Entity, user_rels: list,
                   eid: dict) -> list[QASample]:
        samples = []
        # Count by predicate
        pred_counts = {}
        for rel in user_rels:
            if rel.predicate in self.FAMILY_RELS:
                pred_counts.setdefault(rel.predicate, 0)
                pred_counts[rel.predicate] += 1

        for pred, count in pred_counts.items():
            plural = self.RELATION_PLURALS.get(pred, pred + "s")
            for tmpl in self.COUNT_TEMPLATES:
                samples.append(QASample(
                    question=tmpl.format(rel_plural=plural),
                    answer_mode=2,
                    entity_answer=-1,
                    boolean_answer=-1,
                    count_answer=min(count, 10),
                    hops=1,
                    source_triples=[],
                ))

        # Zero counts for missing relations
        for pred in self.FAMILY_RELS - set(pred_counts.keys()):
            plural = self.RELATION_PLURALS.get(pred, pred + "s")
            for tmpl in self.COUNT_TEMPLATES[:1]:  # fewer negatives
                samples.append(QASample(
                    question=tmpl.format(rel_plural=plural),
                    answer_mode=2,
                    entity_answer=-1,
                    boolean_answer=-1,
                    count_answer=0,
                    hops=1,
                    source_triples=[],
                ))

        # Balance: oversample counts > 1 to match count=1 frequency
        count_1_samples = sum(1 for s in samples if s.count_answer == 1)
        high_count = [s for s in samples if s.count_answer > 1]
        if high_count and count_1_samples > len(high_count):
            repeats = count_1_samples // len(high_count)
            for _ in range(repeats - 1):
                samples.extend(high_count)

        return samples

    def _gen_multi_hop(self, user: Entity, user_rels: list,
                       all_rels: list, eid: dict) -> list[QASample]:
        samples = []

        # For each user relation, find outgoing relations from the target
        for rel1 in user_rels:
            target = eid.get(rel1.object_id)
            if not target:
                continue
            # Find relations FROM this target
            target_rels = [r for r in all_rels if r.subject_id == target.id
                           and r.object_id != user.id]  # skip back-edges

            for rel2 in target_rels:
                final = eid.get(rel2.object_id)
                if not final:
                    continue
                pointer = self.vocab.entity_id_to_pointer(final.id)
                if pointer < 0:
                    continue

                for tmpl in self.MULTI_HOP_TEMPLATES:
                    q = tmpl.format(rel1=rel1.predicate, rel2=rel2.predicate)
                    samples.append(QASample(
                        question=q,
                        answer_mode=0,
                        entity_answer=pointer,
                        boolean_answer=-1,
                        count_answer=-1,
                        hops=2,
                        source_triples=[
                            (user.name, rel1.predicate, target.name),
                            (target.name, rel2.predicate, final.name),
                        ],
                    ))
        return samples

    def _gen_inverse(self, user: Entity, all_rels: list,
                     eid: dict) -> list[QASample]:
        samples = []

        for rel in all_rels:
            subj = eid.get(rel.subject_id)
            obj = eid.get(rel.object_id)
            if not subj or not obj:
                continue
            pointer = self.vocab.entity_id_to_pointer(subj.id)
            if pointer < 0:
                continue

            for tmpl in self.INVERSE_TEMPLATES:
                q = tmpl.format(rel=rel.predicate.replace("_", " "), name=obj.name)
                samples.append(QASample(
                    question=q,
                    answer_mode=0,
                    entity_answer=pointer,
                    boolean_answer=-1,
                    count_answer=-1,
                    hops=1,
                    source_triples=[(subj.name, rel.predicate, obj.name)],
                ))
        return samples


# ─── Training Pipeline ───────────────────────────────────────────

def encode_sample(sample: QASample, vocab: GraphVocab, graph: GraphStore,
                  max_seq_len: int = 128) -> dict:
    """Encode a QASample into model input tensors."""
    cls_idx = vocab.token_to_idx["[CLS]"]
    sep_idx = vocab.token_to_idx["[SEP]"]
    pad_idx = vocab.token_to_idx["[PAD]"]

    # Encode question
    q_tokens = [cls_idx] + vocab.tokenize(sample.question) + [sep_idx]

    # Encode all graph triples
    entities = graph.get_all_entities()
    eid = {e.id: e for e in entities}
    all_rels = graph.get_all_relations(include_inferred=True)

    triple_tokens = []
    for rel in all_rels:
        subj = eid.get(rel.subject_id)
        obj = eid.get(rel.object_id)
        if subj and obj:
            triple_tokens.extend(vocab.encode_triple(subj.name, rel.predicate, obj.name))

    # Combine and pad
    all_tokens = q_tokens + triple_tokens
    if len(all_tokens) > max_seq_len:
        all_tokens = all_tokens[:max_seq_len]

    attention_mask = [1] * len(all_tokens)
    padding = max_seq_len - len(all_tokens)
    all_tokens += [pad_idx] * padding
    attention_mask += [0] * padding

    return {
        "input_ids": torch.tensor(all_tokens, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_mode": torch.tensor(sample.answer_mode, dtype=torch.long),
        "entity_answer": torch.tensor(sample.entity_answer, dtype=torch.long),
        "boolean_answer": torch.tensor(sample.boolean_answer, dtype=torch.float),
        "count_answer": torch.tensor(sample.count_answer, dtype=torch.long),
    }


class ReasoningTrainer:
    """Train the GraphTransformer on synthetic QA data."""

    def __init__(self, model: GraphTransformer, vocab: GraphVocab,
                 graph: GraphStore, user_name: str = "Dimuthu"):
        self.model = model
        self.vocab = vocab
        self.graph = graph
        self.user_name = user_name

    def train(self, epochs: int = 100, lr: float = 0.001,
              eval_split: float = 0.2) -> dict:
        """
        Generate synthetic data, train, and evaluate.
        Returns training history.
        """
        # Seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate data
        gen = SyntheticQAGenerator(self.graph, self.vocab, self.user_name)
        samples = gen.generate_all()
        if len(samples) < 5:
            logger.warning(f"Too few samples ({len(samples)}), skipping training")
            return {"samples": len(samples), "error": "too_few_samples"}

        # Balance classes by oversampling underrepresented modes
        mode_groups = {0: [], 1: [], 2: []}
        for s in samples:
            mode_groups[s.answer_mode].append(s)
        max_count = max(len(g) for g in mode_groups.values())
        balanced = []
        for mode, group in mode_groups.items():
            if group:
                # Oversample to match the largest group
                repeats = max_count // len(group)
                remainder = max_count % len(group)
                balanced.extend(group * repeats + group[:remainder])
        samples = balanced

        print(f"    Generated {len(samples)} QA pairs (balanced)")
        mode_counts = {0: 0, 1: 0, 2: 0}
        for s in samples:
            mode_counts[s.answer_mode] += 1
        print(f"    Entity: {mode_counts[0]}, Boolean: {mode_counts[1]}, Count: {mode_counts[2]}")

        # Encode all samples
        encoded = [encode_sample(s, self.vocab, self.graph) for s in samples]

        # Split train/eval
        n_eval = max(1, int(len(encoded) * eval_split))
        random.shuffle(encoded)
        eval_data = encoded[:n_eval]
        train_data = encoded[n_eval:]

        print(f"    Train: {len(train_data)}, Eval: {len(eval_data)}")

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mode_criterion = nn.CrossEntropyLoss()
        entity_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        boolean_criterion = nn.BCEWithLogitsLoss()
        count_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        history = {"loss": [], "eval_acc": []}
        best_acc = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            random.shuffle(train_data)

            total_loss = 0
            n_batches = 0

            # Mini-batches of 16
            batch_size = min(16, len(train_data))
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                input_ids = torch.stack([b["input_ids"] for b in batch])
                attn_mask = torch.stack([b["attention_mask"] for b in batch])
                modes = torch.stack([b["answer_mode"] for b in batch])
                entities = torch.stack([b["entity_answer"] for b in batch])
                booleans = torch.stack([b["boolean_answer"] for b in batch])
                counts = torch.stack([b["count_answer"] for b in batch])

                optimizer.zero_grad()
                out = self.model(input_ids, attn_mask)

                # Mode loss (always)
                loss = mode_criterion(out["mode_logits"], modes)

                # Entity pointer loss (only for entity-mode samples)
                ent_mask = (modes == 0) & (entities >= 0)
                if ent_mask.any():
                    loss += entity_criterion(out["entity_logits"][ent_mask],
                                             entities[ent_mask])

                # Boolean loss (only for boolean-mode samples)
                bool_mask = (modes == 1) & (booleans >= 0)
                if bool_mask.any():
                    loss += boolean_criterion(out["boolean_logits"][bool_mask].squeeze(-1),
                                              booleans[bool_mask])

                # Count loss (only for count-mode samples)
                count_mask = (modes == 2) & (counts >= 0)
                if count_mask.any():
                    count_targets = counts[count_mask].clamp(0, 5)  # cap at 5
                    loss += count_criterion(out["count_logits"][count_mask],
                                            count_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            # Eval every 10 epochs
            if (epoch + 1) % 10 == 0:
                acc = self._evaluate(eval_data)
                history["eval_acc"].append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                if (epoch + 1) % 25 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, eval_acc={acc:.1%}")

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        return {
            "samples": len(samples),
            "train_size": len(train_data),
            "eval_size": len(eval_data),
            "final_loss": history["loss"][-1],
            "best_eval_acc": best_acc,
            "epochs": epochs,
        }

    def _evaluate(self, eval_data: list) -> float:
        """Evaluate on held-out data. Returns accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for sample in eval_data:
                input_ids = sample["input_ids"].unsqueeze(0)
                attn_mask = sample["attention_mask"].unsqueeze(0)
                out = self.model(input_ids, attn_mask)

                # Check mode prediction
                pred_mode = out["mode_logits"].argmax(dim=-1).item()
                true_mode = sample["answer_mode"].item()

                if pred_mode != true_mode:
                    total += 1
                    continue

                # Check mode-specific answer
                if true_mode == 0:  # entity
                    pred_ent = out["entity_logits"].argmax(dim=-1).item()
                    true_ent = sample["entity_answer"].item()
                    if pred_ent == true_ent:
                        correct += 1
                elif true_mode == 1:  # boolean
                    pred_bool = (out["boolean_logits"].squeeze() > 0).int().item()
                    true_bool = int(sample["boolean_answer"].item())
                    if pred_bool == true_bool:
                        correct += 1
                elif true_mode == 2:  # count
                    pred_count = out["count_logits"].argmax(dim=-1).item()
                    true_count = sample["count_answer"].item()
                    if pred_count == true_count:
                        correct += 1
                total += 1

        return correct / max(total, 1)


# ─── Top-Level Reasoner ─────────────────────────────────────────

@dataclass
class ReasoningResult:
    """Result of a reasoning query."""
    answered: bool = False
    answer: str = ""
    confidence: float = 0.0
    answer_mode: str = ""    # "entity", "boolean", "count"
    raw_entity: str = ""     # entity name if entity mode
    raw_boolean: bool = False
    raw_count: int = 0
    reasoning_trace: str = ""  # human-readable explanation
    source: str = "reasoner"


class GraphReasoner:
    """
    Top-level interface for the micro-transformer reasoner.

    Usage:
        reasoner = GraphReasoner(graph, user_name="Dimuthu")
        reasoner.train(epochs=100)
        result = reasoner.reason("Who is my wife's father?")
    """

    def __init__(self, graph: GraphStore, user_name: str = "Dimuthu",
                 model_dir: str = "data/reasoner"):
        self.graph = graph
        self.user_name = user_name
        self.model_dir = model_dir
        self.vocab = GraphVocab()
        self.model = None
        self._trained = False

    def train(self, epochs: int = 100, lr: float = 0.001) -> dict:
        """Build vocab, create model, generate data, train."""
        t0 = time.time()

        # Build vocabulary from current graph
        self.vocab.build(self.graph)
        print(f"    Vocab: {self.vocab.size} tokens, {self.vocab.n_entities} entities")

        # Create model
        self.model = GraphTransformer(
            vocab_size=self.vocab.size,
            n_entities=self.vocab.n_entities,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.3,
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    Model: {total_params:,} parameters")

        # Train
        trainer = ReasoningTrainer(self.model, self.vocab, self.graph, self.user_name)
        results = trainer.train(epochs=epochs, lr=lr)

        # Save
        self._save()
        results["duration_ms"] = (time.time() - t0) * 1000
        results["params"] = total_params
        self._trained = True

        return results

    def reason(self, question: str) -> ReasoningResult:
        """
        Attempt to answer a question using the micro-transformer.
        Returns ReasoningResult with answer and confidence.
        """
        if not self._trained or self.model is None:
            if not self._load():
                return ReasoningResult(answered=False, reasoning_trace="Model not trained")

        self.model.eval()

        # Encode question + all triples
        sample = QASample(
            question=question,
            answer_mode=0, entity_answer=-1,
            boolean_answer=-1, count_answer=-1,
            hops=0, source_triples=[],
        )
        encoded = encode_sample(sample, self.vocab, self.graph)

        with torch.no_grad():
            out = self.model(
                encoded["input_ids"].unsqueeze(0),
                encoded["attention_mask"].unsqueeze(0),
            )

        # Determine answer mode
        mode_probs = F.softmax(out["mode_logits"], dim=-1).squeeze()
        pred_mode = mode_probs.argmax().item()
        mode_confidence = mode_probs[pred_mode].item()

        mode_names = ["entity", "boolean", "count"]
        trace_parts = [f"Mode: {mode_names[pred_mode]} (conf={mode_confidence:.2f})"]

        result = ReasoningResult(source="reasoner")

        if pred_mode == 0:  # Entity pointer
            ent_probs = F.softmax(out["entity_logits"], dim=-1).squeeze()
            pred_ent = ent_probs.argmax().item()
            ent_confidence = ent_probs[pred_ent].item()

            entity_name = self.vocab.pointer_to_entity_name(pred_ent)
            result.answer_mode = "entity"
            result.raw_entity = entity_name
            result.answer = entity_name.title()
            result.confidence = mode_confidence * ent_confidence
            trace_parts.append(f"Entity: {entity_name} (conf={ent_confidence:.2f})")

            # Show top-3 entity candidates
            top3 = ent_probs.topk(min(3, len(ent_probs)))
            for idx, prob in zip(top3.indices, top3.values):
                name = self.vocab.pointer_to_entity_name(idx.item())
                trace_parts.append(f"  candidate: {name} ({prob.item():.2f})")

        elif pred_mode == 1:  # Boolean
            bool_prob = torch.sigmoid(out["boolean_logits"]).squeeze().item()
            pred_bool = bool_prob > 0.5
            bool_confidence = bool_prob if pred_bool else (1 - bool_prob)

            result.answer_mode = "boolean"
            result.raw_boolean = pred_bool
            result.answer = "Yes" if pred_bool else "No"
            result.confidence = mode_confidence * bool_confidence
            trace_parts.append(f"Boolean: {'Yes' if pred_bool else 'No'} (conf={bool_confidence:.2f})")

        elif pred_mode == 2:  # Count (classification 0-5)
            count_probs = F.softmax(out["count_logits"], dim=-1).squeeze()
            pred_count = count_probs.argmax().item()
            count_confidence = count_probs[pred_count].item()

            result.answer_mode = "count"
            result.raw_count = pred_count
            result.answer = str(pred_count)
            result.confidence = mode_confidence * count_confidence
            trace_parts.append(f"Count: {pred_count} (conf={count_confidence:.2f})")

        result.answered = result.confidence > 0.3  # low threshold; caller decides fallback
        result.reasoning_trace = " | ".join(trace_parts)

        return result

    def _save(self):
        """Save model + vocab to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        if self.model:
            torch.save({
                "model_state": self.model.state_dict(),
                "vocab": {
                    "token_to_idx": self.vocab.token_to_idx,
                    "idx_to_token": {int(k): v for k, v in self.vocab.idx_to_token.items()},
                    "n_entities": self.vocab.n_entities,
                    "entity_id_to_pointer": self.vocab._entity_id_to_pointer,
                    "pointer_to_entity_name": self.vocab._pointer_to_entity_name,
                    "entity_id_to_vocab": self.vocab._entity_id_to_vocab,
                },
                "config": {
                    "vocab_size": self.vocab.size,
                    "n_entities": self.vocab.n_entities,
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                },
            }, os.path.join(self.model_dir, "reasoner.pt"))

    def _load(self) -> bool:
        """Load model + vocab from disk."""
        path = os.path.join(self.model_dir, "reasoner.pt")
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            cfg = checkpoint["config"]

            # Restore vocab
            v = checkpoint["vocab"]
            self.vocab.token_to_idx = v["token_to_idx"]
            self.vocab.idx_to_token = {int(k): val for k, val in v["idx_to_token"].items()}
            self.vocab.n_entities = v["n_entities"]
            self.vocab._entity_id_to_pointer = v.get("entity_id_to_pointer", {})
            self.vocab._pointer_to_entity_name = {int(k): val for k, val in v.get("pointer_to_entity_name", {}).items()}
            self.vocab._entity_id_to_vocab = v.get("entity_id_to_vocab", {})

            # Restore model
            self.model = GraphTransformer(
                vocab_size=cfg["vocab_size"],
                n_entities=cfg["n_entities"],
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                dropout=0.0,  # no dropout at inference
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self._trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load reasoner: {e}")
            return False
