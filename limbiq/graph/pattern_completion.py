"""
Pattern Completion Model for Knowledge Graph — Phase 3
=======================================================
Three components:

1. Entity Resolution — merge duplicate entities (e.g., "default" → "Dimuthu")
2. Link Prediction — TransE-style knowledge graph embeddings that learn
   (subject, relation, object) triple patterns and predict missing links
3. Learned Inference — replaces the buggy hand-written inference rules with
   a trained model that predicts implied relations from observed patterns

Architecture: TransE with type constraints
  score(h, r, t) = -||h + r - t||
  For each (s, r, ?) predict top-k tail entities
  For each (s, ?, o) predict relation type

Total params: ~50K (tiny, trains in seconds)
"""

import os
import re
import json
import time
import math
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from limbiq.graph.store import GraphStore, Entity, Relation

logger = logging.getLogger(__name__)


# ─── Entity Resolution ──────────────────────────────────────────

class EntityResolver:
    """
    Merge duplicate entities in the graph.

    Primary job: resolve "default" (limbiq's internal user_id) into
    the actual user name, reassigning all relations.
    Also handles other duplicate entities (case variants, etc.)
    """

    def __init__(self, graph: GraphStore, user_name: str = "Dimuthu"):
        self.graph = graph
        self.user_name = user_name

    def resolve(self) -> dict:
        """Run full entity resolution. Returns stats."""
        stats = {"merged": 0, "relations_reassigned": 0, "duplicates_removed": 0}

        # 1. Merge "default" → user_name
        stats.update(self._merge_default_to_user())

        # 2. Merge case-insensitive duplicates
        stats.update(self._merge_case_duplicates())

        # 3. Remove orphaned entities (no relations)
        stats["orphans_removed"] = self._remove_orphans()

        return stats

    def _merge_default_to_user(self) -> dict:
        """Reassign all 'default' entity relations to the actual user entity."""
        default_ent = self.graph.find_entity_by_name("default")
        user_ent = self.graph.find_entity_by_name(self.user_name)

        if not default_ent or not user_ent:
            return {"merged": 0, "relations_reassigned": 0}

        # Get all relations involving "default"
        relations = self.graph.get_relations_for(default_ent.id)
        reassigned = 0

        for rel in relations:
            new_subj = user_ent.id if rel.subject_id == default_ent.id else rel.subject_id
            new_obj = user_ent.id if rel.object_id == default_ent.id else rel.object_id

            # Check if this relation already exists for the user
            existing = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (new_subj, rel.predicate, new_obj)
            ).fetchone()

            if existing:
                # Delete the duplicate default relation
                self.graph.db.execute("DELETE FROM relations WHERE id=?", (rel.id,))
            else:
                # Reassign to user entity
                self.graph.db.execute(
                    "UPDATE relations SET subject_id=? WHERE id=? AND subject_id=?",
                    (user_ent.id, rel.id, default_ent.id)
                )
                self.graph.db.execute(
                    "UPDATE relations SET object_id=? WHERE id=? AND object_id=?",
                    (user_ent.id, rel.id, default_ent.id)
                )
                reassigned += 1

        # Delete the "default" entity itself
        self.graph.db.execute("DELETE FROM entities WHERE id=?", (default_ent.id,))
        self.graph.db.commit()

        return {"merged": 1, "relations_reassigned": reassigned}

    def _merge_case_duplicates(self) -> dict:
        """Merge entities that differ only by case."""
        entities = self.graph.get_all_entities()
        name_groups = defaultdict(list)
        for e in entities:
            name_groups[e.name.lower()].append(e)

        duplicates_removed = 0
        for name, group in name_groups.items():
            if len(group) < 2:
                continue

            # Keep the one with the most relations
            rel_counts = []
            for e in group:
                count = len(self.graph.get_relations_for(e.id))
                rel_counts.append((count, e))
            rel_counts.sort(key=lambda x: -x[0])

            keep = rel_counts[0][1]
            for _, dup in rel_counts[1:]:
                # Reassign relations
                self.graph.db.execute(
                    "UPDATE relations SET subject_id=? WHERE subject_id=?",
                    (keep.id, dup.id)
                )
                self.graph.db.execute(
                    "UPDATE relations SET object_id=? WHERE object_id=?",
                    (keep.id, dup.id)
                )
                self.graph.db.execute("DELETE FROM entities WHERE id=?", (dup.id,))
                duplicates_removed += 1

        self.graph.db.commit()
        return {"duplicates_removed": duplicates_removed}

    def _remove_orphans(self) -> int:
        """Remove entities with zero relations (except the user)."""
        entities = self.graph.get_all_entities()
        removed = 0
        for e in entities:
            if e.name.lower() == self.user_name.lower():
                continue
            rels = self.graph.get_relations_for(e.id)
            if len(rels) == 0:
                self.graph.db.execute("DELETE FROM entities WHERE id=?", (e.id,))
                removed += 1
        self.graph.db.commit()
        return removed


# ─── Graph Cleanup ──────────────────────────────────────────────

class GraphCleanup:
    """
    Clean up data quality issues in the graph:
    - Contradictory relations (e.g., Dimuthu→husband→Prabhashi + Dimuthu→wife→Prabhashi)
    - Garbage entities from misparses (e.g., "wifes", "your father", "you")
    - Predicate normalization (e.g., "father-in-law" → "father_in_law_of")
    """

    # Entities that are clearly misparses and should be removed
    GARBAGE_ENTITY_PATTERNS = [
        r"^your\s+",      # "your father", "your mother"
        r"^my\s+",        # "my wife", "my father"
        r"^the\s+",       # "the user"
        r"^wifes?$",      # misparse of "wife's"
        r"^husbands?$",   # misparse of "husband's"
        r"^you$",         # pronoun
        r"^me$",          # pronoun
        r"^user$",        # generic
        r"^concept$",     # noise
        r"'s\s+\w+$",     # "Prabhashi's father"
    ]

    # Predicate normalization: map variant forms to canonical
    PREDICATE_NORMALIZE = {
        "father-in-law": "father_in_law_of",
        "mother-in-law": "mother_in_law_of",
        "brother-in-law": "brother_in_law_of",
        "sister-in-law": "sister_in_law_of",
        "father in law": "father_in_law_of",
        "mother in law": "mother_in_law_of",
    }

    # Contradictory relation pairs: (predA, predB) on same (subject, object) — remove predB
    # e.g., if Dimuthu→wife→X AND Dimuthu→husband→X, remove husband (keep wife)
    CONTRADICTORY_PAIRS = [
        ("wife", "husband"),     # if X→wife→Y, then X→husband→Y is wrong (X is the husband, not Y)
    ]

    # Spouse relation normalization: ensure consistent direction
    # If A→wife→B exists, then B→husband→A is correct, but B→wife→A is wrong
    SPOUSE_PAIRS = {
        "wife": "husband",   # if A→wife→B, then B should have husband→A (not wife→A)
        "husband": "wife",   # if A→husband→B, then B should have wife→A (not husband→A)
    }

    def __init__(self, graph: GraphStore, user_name: str = "Dimuthu"):
        self.graph = graph
        self.user_name = user_name

    def cleanup(self) -> dict:
        """Run all cleanup steps. Returns stats."""
        stats = {}
        stats["garbage_entities_removed"] = self._remove_garbage_entities()
        stats["predicates_normalized"] = self._normalize_predicates()
        stats["contradictions_fixed"] = self._fix_contradictions()
        stats["spouse_direction_fixed"] = self._fix_spouse_directions()
        stats["dangling_relations_removed"] = self._remove_dangling_relations()
        return stats

    def _remove_garbage_entities(self) -> int:
        """Remove entities that are clearly misparses."""
        import re as _re
        entities = self.graph.get_all_entities()
        removed = 0
        for e in entities:
            for pattern in self.GARBAGE_ENTITY_PATTERNS:
                if _re.search(pattern, e.name, _re.IGNORECASE):
                    # Delete all relations involving this entity first
                    self.graph.db.execute(
                        "DELETE FROM relations WHERE subject_id=? OR object_id=?",
                        (e.id, e.id)
                    )
                    self.graph.db.execute("DELETE FROM entities WHERE id=?", (e.id,))
                    removed += 1
                    logger.info(f"  Removed garbage entity: {e.name}")
                    break
        self.graph.db.commit()
        return removed

    def _normalize_predicates(self) -> int:
        """Normalize predicate names to canonical form."""
        normalized = 0
        for old_pred, new_pred in self.PREDICATE_NORMALIZE.items():
            rows = self.graph.db.execute(
                "SELECT id, subject_id, object_id FROM relations WHERE predicate=?",
                (old_pred,)
            ).fetchall()
            for row in rows:
                rel_id, subj_id, obj_id = row
                # Check if canonical form already exists
                existing = self.graph.db.execute(
                    "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                    (subj_id, new_pred, obj_id)
                ).fetchone()
                if existing:
                    # Duplicate — just delete the old one
                    self.graph.db.execute("DELETE FROM relations WHERE id=?", (rel_id,))
                else:
                    self.graph.db.execute(
                        "UPDATE relations SET predicate=? WHERE id=?",
                        (new_pred, rel_id)
                    )
                normalized += 1
                logger.info(f"  Normalized predicate: {old_pred} → {new_pred}")
        self.graph.db.commit()
        return normalized

    def _fix_contradictions(self) -> int:
        """Remove contradictory relations."""
        fixed = 0
        for keep_pred, remove_pred in self.CONTRADICTORY_PAIRS:
            # Find cases where both exist on same (subject, object)
            rows = self.graph.db.execute(
                """SELECT r1.id as keep_id, r2.id as remove_id,
                          r1.subject_id, r1.object_id
                   FROM relations r1
                   JOIN relations r2 ON r1.subject_id = r2.subject_id
                                    AND r1.object_id = r2.object_id
                   WHERE r1.predicate=? AND r2.predicate=?""",
                (keep_pred, remove_pred)
            ).fetchall()
            for row in rows:
                self.graph.db.execute("DELETE FROM relations WHERE id=?", (row[1],))
                fixed += 1
                logger.info(f"  Removed contradictory: kept {keep_pred}, removed {remove_pred}")
        self.graph.db.commit()
        return fixed

    def _fix_spouse_directions(self) -> int:
        """
        Fix spouse relation directions.

        Key insight: the user entity should be the SUBJECT of wife/husband
        relations. If Dimuthu→wife→Prabhashi exists, that's canonical.
        Prabhashi→wife→Dimuthu is wrong (Prabhashi is not Dimuthu's wife in
        the other direction — she's his wife, he's her husband).

        Strategy:
        1. Find the user entity
        2. For user's spouse relations, ensure only correct directions exist
        3. Ensure the reverse relation uses the correct predicate
        """
        user_ent = self.graph.find_entity_by_name(self.user_name)
        if not user_ent:
            return 0

        fixed = 0
        all_rels = self.graph.get_all_relations(include_inferred=False)

        # First pass: identify user's canonical spouse relations
        user_spouse_rels = [r for r in all_rels
                           if r.subject_id == user_ent.id
                           and r.predicate in self.SPOUSE_PAIRS]

        for rel in user_spouse_rels:
            spouse_id = rel.object_id
            expected_reverse = self.SPOUSE_PAIRS[rel.predicate]

            # Remove same predicate in reverse (e.g., Prabhashi→wife→Dimuthu)
            rows = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (spouse_id, rel.predicate, user_ent.id)
            ).fetchall()
            for row in rows:
                self.graph.db.execute("DELETE FROM relations WHERE id=?", (row[0],))
                fixed += 1

            # Remove same direction opposite predicate (e.g., Dimuthu→husband→Prabhashi)
            rows = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (user_ent.id, expected_reverse, spouse_id)
            ).fetchall()
            for row in rows:
                self.graph.db.execute("DELETE FROM relations WHERE id=?", (row[0],))
                fixed += 1

            # Ensure correct reverse exists (e.g., Prabhashi→husband→Dimuthu)
            existing_reverse = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (spouse_id, expected_reverse, user_ent.id)
            ).fetchone()
            if not existing_reverse:
                from limbiq.graph.store import Relation as _Rel
                self.graph.add_relation(_Rel(
                    subject_id=spouse_id,
                    predicate=expected_reverse,
                    object_id=user_ent.id,
                    confidence=rel.confidence,
                    is_inferred=False,
                ))

        # Also handle non-user spouse pairs (e.g., Dilini↔Dilanka)
        # For these, just ensure no mirror duplicates of the SAME predicate
        non_user_spouse = [r for r in all_rels
                          if r.subject_id != user_ent.id
                          and r.object_id != user_ent.id
                          and r.predicate in self.SPOUSE_PAIRS]

        seen_pairs = set()
        for rel in non_user_spouse:
            pair = (min(rel.subject_id, rel.object_id), max(rel.subject_id, rel.object_id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Check for same predicate in both directions
            mirror = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (rel.object_id, rel.predicate, rel.subject_id)
            ).fetchone()
            if mirror:
                self.graph.db.execute("DELETE FROM relations WHERE id=?", (mirror[0],))
                fixed += 1

        self.graph.db.commit()
        return fixed

    def _remove_dangling_relations(self) -> int:
        """Remove relations that reference non-existent entities."""
        removed = self.graph.db.execute(
            """DELETE FROM relations WHERE
               subject_id NOT IN (SELECT id FROM entities)
               OR object_id NOT IN (SELECT id FROM entities)"""
        ).rowcount
        self.graph.db.commit()
        return removed


# ─── Memory-Based Relation Extraction ────────────────────────────

class RelationMiner:
    """
    Mine additional relations from existing memories that the original
    EntityExtractor missed. Uses more aggressive pattern matching on
    compressed/reformatted memory content.
    """

    # Patterns for family relations in various memory formats
    FAMILY_PATTERNS = [
        # Direct statements
        (r"(?:my|user'?s?)\s+mother(?:'s name)?\s+is\s+(\w+)", "mother"),
        (r"(\w+)\s+is\s+(?:my|the user'?s?)\s+mother", "mother"),
        (r"mother\s*[:=]\s*(\w+)", "mother"),
        # Sisters
        (r"(?:my|user'?s?)\s+(?:elder\s+)?sister(?:'s name)?\s+is\s+(\w+)", "sister"),
        (r"(?:my|user'?s?)\s+(?:younger\s+)?sister(?:'s name)?\s+is\s+(\w+)", "sister"),
        (r"(\w+)\s+is\s+(?:my|the user'?s?)\s+(?:elder\s+|younger\s+)?sister", "sister"),
        (r"sisters?\s*[:=]\s*(\w+)", "sister"),
        (r"(?:two|2)\s+sisters?\b.*?(\w+)\s+and\s+(\w+)", "sister_pair"),
        # Father
        (r"(?:my|user'?s?)\s+father(?:'s name)?\s+is\s+(\w+)", "father"),
        (r"(\w+)\s+is\s+(?:my|the user'?s?)\s+father", "father"),
        # Wife/husband
        (r"(?:my|user'?s?)\s+wife(?:'s name)?\s+is\s+(\w+)", "wife"),
        (r"(\w+)\s+is\s+(?:my|the user'?s?)\s+wife", "wife"),
        # Marriage
        (r"(\w+)\s+(?:is\s+)?married\s+to\s+(\w+)", "married"),
    ]

    # Known entity names to validate extractions
    KNOWN_ENTITIES = {"Dimuthu", "Prabhashi", "Upananda", "Renuka", "Dilini", "Dilani", "Dilanka"}

    def __init__(self, graph: GraphStore, store_db, user_name: str = "Dimuthu"):
        self.graph = graph
        self.store_db = store_db
        self.user_name = user_name

    def mine_relations(self) -> dict:
        """Mine all memories for missing relations. Returns stats."""
        # Get all non-suppressed memories
        rows = self.store_db.execute(
            "SELECT id, content FROM memories WHERE is_suppressed = 0"
        ).fetchall()

        new_relations = 0
        user_entity = self.graph.find_entity_by_name(self.user_name)
        if not user_entity:
            return {"new_relations": 0}

        found_relations = set()  # (subject, predicate, object) to deduplicate

        for mem_id, content in rows:
            for pattern, rel_type in self.FAMILY_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    if rel_type == "sister_pair":
                        names = [match.group(1), match.group(2)]
                        for name in names:
                            if name in self.KNOWN_ENTITIES:
                                found_relations.add((self.user_name, "sister", name, mem_id))
                    elif rel_type == "married":
                        name_a, name_b = match.group(1), match.group(2)
                        if name_a in self.KNOWN_ENTITIES and name_b in self.KNOWN_ENTITIES:
                            found_relations.add((name_a, "husband", name_b, mem_id))
                            found_relations.add((name_b, "wife", name_a, mem_id))
                    else:
                        name = match.group(1)
                        if name in self.KNOWN_ENTITIES:
                            found_relations.add((self.user_name, rel_type, name, mem_id))

        # Also scan suppressed memories for family facts (they may have been
        # incorrectly suppressed but still contain valid information)
        rows_suppressed = self.store_db.execute(
            "SELECT id, content FROM memories WHERE is_suppressed = 1"
        ).fetchall()

        for mem_id, content in rows_suppressed:
            for pattern, rel_type in self.FAMILY_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    if rel_type == "sister_pair":
                        names = [match.group(1), match.group(2)]
                        for name in names:
                            if name in self.KNOWN_ENTITIES:
                                found_relations.add((self.user_name, "sister", name, mem_id))
                    elif rel_type == "married":
                        name_a, name_b = match.group(1), match.group(2)
                        if name_a in self.KNOWN_ENTITIES and name_b in self.KNOWN_ENTITIES:
                            found_relations.add((name_a, "husband", name_b, mem_id))
                            found_relations.add((name_b, "wife", name_a, mem_id))
                    elif rel_type in ("mother", "father", "sister", "wife"):
                        name = match.group(1)
                        if name in self.KNOWN_ENTITIES:
                            found_relations.add((self.user_name, rel_type, name, mem_id))

        # Add found relations to graph
        for subj_name, pred, obj_name, source_mem in found_relations:
            subj = self.graph.find_entity_by_name(subj_name)
            obj = self.graph.find_entity_by_name(obj_name)
            if not subj or not obj:
                continue

            # Check if already exists
            existing = self.graph.db.execute(
                "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (subj.id, pred, obj.id)
            ).fetchone()
            if not existing:
                self.graph.add_relation(Relation(
                    subject_id=subj.id,
                    predicate=pred,
                    object_id=obj.id,
                    confidence=0.85,
                    is_inferred=False,
                    source_memory_id=source_mem,
                ))
                new_relations += 1
                logger.info(f"  Mined: {subj_name} --[{pred}]--> {obj_name}")

        return {"new_relations": new_relations}


# ─── TransE Knowledge Graph Embeddings ───────────────────────────

class TransE(nn.Module):
    """
    TransE: Translation-based knowledge graph embedding.

    Core idea: h + r ≈ t for true triples (h, r, t)

    Learns embedding vectors for entities and relations such that
    the translation score ||h + r - t|| is low for true triples
    and high for corrupted triples.

    Used for:
    - Link prediction: given (h, r, ?), rank all possible tails
    - Relation prediction: given (h, ?, t), rank all possible relations
    - Triple validation: score arbitrary (h, r, t) triples
    """

    def __init__(self, n_entities: int, n_relations: int, dim: int = 64, margin: float = 1.0):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.n_entities = n_entities
        self.n_relations = n_relations

        # Entity and relation embeddings
        self.entity_emb = nn.Embedding(n_entities, dim)
        self.relation_emb = nn.Embedding(n_relations, dim)

        # Initialize with uniform distribution
        nn.init.uniform_(self.entity_emb.weight, -6.0/math.sqrt(dim), 6.0/math.sqrt(dim))
        nn.init.uniform_(self.relation_emb.weight, -6.0/math.sqrt(dim), 6.0/math.sqrt(dim))

        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, dim=-1)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """
        Compute TransE scores for triples.
        Lower score = more plausible triple.
        """
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        # L2 distance: ||h + r - t||
        return (h + r - t).norm(p=2, dim=-1)

    def predict_tails(self, head: int, relation: int, top_k: int = 5) -> list:
        """Given (h, r, ?), predict top-k tail entities."""
        h = self.entity_emb.weight[head]
        r = self.relation_emb.weight[relation]

        # h + r should be close to t
        target = h + r
        distances = (self.entity_emb.weight - target.unsqueeze(0)).norm(p=2, dim=-1)

        top_k = min(top_k, self.n_entities)
        values, indices = distances.topk(top_k, largest=False)
        return [(idx.item(), -val.item()) for idx, val in zip(indices, values)]

    def predict_relations(self, head: int, tail: int, top_k: int = 5) -> list:
        """Given (h, ?, t), predict top-k relations."""
        h = self.entity_emb.weight[head]
        t = self.entity_emb.weight[tail]

        # r should be close to t - h
        target = t - h
        distances = (self.relation_emb.weight - target.unsqueeze(0)).norm(p=2, dim=-1)

        top_k = min(top_k, self.n_relations)
        values, indices = distances.topk(top_k, largest=False)
        return [(idx.item(), -val.item()) for idx, val in zip(indices, values)]

    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score a single triple. Lower = more plausible."""
        h = self.entity_emb.weight[head]
        r = self.relation_emb.weight[relation]
        t = self.entity_emb.weight[tail]
        return (h + r - t).norm(p=2).item()


# ─── Training Pipeline ──────────────────────────────────────────

def build_kg_data(graph: GraphStore) -> dict:
    """
    Build entity/relation indices and training triples from the graph.
    """
    entities = graph.get_all_entities()
    relations = graph.get_all_relations(include_inferred=False)

    # Build indices
    entity_to_idx = {}
    idx_to_entity = {}
    for i, e in enumerate(entities):
        entity_to_idx[e.id] = i
        idx_to_entity[i] = e

    # Collect unique predicate types
    predicates = sorted(set(r.predicate for r in relations))
    pred_to_idx = {p: i for i, p in enumerate(predicates)}
    idx_to_pred = {i: p for p, i in pred_to_idx.items()}

    # Build triples
    triples = []
    for r in relations:
        if r.subject_id in entity_to_idx and r.object_id in entity_to_idx:
            h = entity_to_idx[r.subject_id]
            rel = pred_to_idx[r.predicate]
            t = entity_to_idx[r.object_id]
            triples.append((h, rel, t))

    return {
        "entity_to_idx": entity_to_idx,
        "idx_to_entity": idx_to_entity,
        "pred_to_idx": pred_to_idx,
        "idx_to_pred": idx_to_pred,
        "triples": triples,
        "n_entities": len(entities),
        "n_relations": len(predicates),
    }


def generate_negative_triples(triples: list, n_entities: int,
                              neg_ratio: int = 5) -> list:
    """Generate corrupted triples for contrastive training."""
    negatives = []
    triple_set = set(triples)

    for h, r, t in triples:
        for _ in range(neg_ratio):
            # Corrupt head or tail (50/50)
            if np.random.random() < 0.5:
                new_h = np.random.randint(0, n_entities)
                while (new_h, r, t) in triple_set:
                    new_h = np.random.randint(0, n_entities)
                negatives.append((new_h, r, t))
            else:
                new_t = np.random.randint(0, n_entities)
                while (h, r, new_t) in triple_set:
                    new_t = np.random.randint(0, n_entities)
                negatives.append((h, r, new_t))

    return negatives


def train_transe(model: TransE, triples: list, n_entities: int,
                 epochs: int = 500, lr: float = 0.01) -> dict:
    """Train TransE with margin-based ranking loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "mrr": []}
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Generate negative samples each epoch (fresh corruption)
        negatives = generate_negative_triples(triples, n_entities, neg_ratio=5)

        pos_h = torch.tensor([t[0] for t in triples])
        pos_r = torch.tensor([t[1] for t in triples])
        pos_t = torch.tensor([t[2] for t in triples])

        neg_h = torch.tensor([t[0] for t in negatives])
        neg_r = torch.tensor([t[1] for t in negatives])
        neg_t = torch.tensor([t[2] for t in negatives])

        optimizer.zero_grad()

        # Positive and negative scores
        pos_scores = model(pos_h, pos_r, pos_t)
        neg_scores = model(neg_h, neg_r, neg_t)

        # Repeat positives to match negative ratio
        pos_expanded = pos_scores.repeat_interleave(5)

        # Margin ranking loss: max(0, margin + pos - neg)
        loss = F.relu(model.margin + pos_expanded - neg_scores).mean()

        # Regularization: keep embeddings on unit sphere
        reg = (model.entity_emb.weight.norm(p=2, dim=-1) - 1).pow(2).mean() * 0.1
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()

        # Normalize entity embeddings
        with torch.no_grad():
            model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, dim=-1)

        history["loss"].append(total_loss.item())

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 100 == 0:
            # Compute MRR on training set
            mrr = compute_mrr(model, triples, n_entities)
            history["mrr"].append(mrr)
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss.item():.4f}, MRR={mrr:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    return history


def compute_mrr(model: TransE, triples: list, n_entities: int) -> float:
    """Compute Mean Reciprocal Rank for tail prediction."""
    if not triples:
        return 0.0

    reciprocal_ranks = []
    with torch.no_grad():
        for h, r, t in triples:
            # Score all possible tails
            heads = torch.full((n_entities,), h, dtype=torch.long)
            rels = torch.full((n_entities,), r, dtype=torch.long)
            tails = torch.arange(n_entities)
            scores = model(heads, rels, tails)

            # Rank (lower score = better)
            sorted_indices = scores.argsort()
            rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)


# ─── Learned Inference Engine ─────────────────────────────────────

class LearnedInference:
    """
    Replace hand-written inference rules with learned TransE predictions.

    For each entity pair (A, B) where no direct relation exists,
    uses TransE to predict whether a relation should exist and what it is.

    Also validates existing inferred relations — removes wrong ones
    (like the grandfather_of bug) and adds correct ones.
    """

    # Relation type constraints: which entity type pairs are valid
    TYPE_CONSTRAINTS = {
        "father_in_law_of": ("person", "person"),
        "mother_in_law_of": ("person", "person"),
        "brother_in_law_of": ("person", "person"),
        "sister_in_law_of": ("person", "person"),
        "grandfather_of": ("person", "person"),
        "grandmother_of": ("person", "person"),
        "uncle_of": ("person", "person"),
        "aunt_of": ("person", "person"),
        "works_at": ("person", "company"),
        "role": ("person", "person"),
    }

    # Semantic constraints: what makes sense given family structure
    # (parent_of_user, spouse_of_user) → parent is in-law of spouse
    SEMANTIC_RULES = {
        ("father", "wife"): "father_in_law_of",     # user's father → wife's father-in-law
        ("mother", "wife"): "mother_in_law_of",      # user's mother → wife's mother-in-law
        ("father", "husband"): "father_in_law_of",
        ("mother", "husband"): "mother_in_law_of",
        ("sister", "wife"): "sister_in_law_of",      # user's sister → wife's sister-in-law
        ("sister", "husband"): "sister_in_law_of",
        ("brother", "wife"): "brother_in_law_of",
        ("brother", "husband"): "brother_in_law_of",
    }

    def __init__(self, graph: GraphStore, model: TransE = None, kg_data: dict = None):
        self.graph = graph
        self.model = model
        self.kg_data = kg_data

    def run_inference(self, user_name: str = "Dimuthu") -> dict:
        """
        Run learned inference. Combines:
        1. Corrected semantic rules (fixes the grandfather bug)
        2. TransE predictions for novel links
        """
        # Clear old inferences
        self.graph.remove_inferred()

        user = self.graph.find_entity_by_name(user_name)
        if not user:
            return {"inferred": 0, "transe_predictions": 0}

        # Get all explicit relations FROM the user
        all_relations = self.graph.get_all_relations(include_inferred=False)
        user_relations = [r for r in all_relations if r.subject_id == user.id]

        inferred_count = 0
        entities_by_id = {e.id: e for e in self.graph.get_all_entities()}

        # Apply corrected semantic rules
        for rel_a in user_relations:
            for rel_b in user_relations:
                if rel_a.id == rel_b.id:
                    continue

                key = (rel_a.predicate, rel_b.predicate)
                inferred_pred = self.SEMANTIC_RULES.get(key)
                if not inferred_pred:
                    continue

                # The inferred relation goes FROM rel_a's object TO rel_b's object
                # e.g., (Dimuthu→father→Upananda) + (Dimuthu→wife→Prabhashi)
                # → Upananda is father_in_law_of Prabhashi
                subj_id = rel_a.object_id
                obj_id = rel_b.object_id

                # Don't create self-relations
                if subj_id == obj_id:
                    continue

                confidence = min(rel_a.confidence, rel_b.confidence) * 0.9

                self.graph.add_relation(Relation(
                    subject_id=subj_id,
                    predicate=inferred_pred,
                    object_id=obj_id,
                    confidence=confidence,
                    is_inferred=True,
                ))
                inferred_count += 1

        # TransE-based link prediction (if model available)
        transe_predictions = 0
        if self.model and self.kg_data:
            transe_predictions = self._transe_inference(user, entities_by_id)

        return {
            "inferred": inferred_count,
            "transe_predictions": transe_predictions,
        }

    # Semantic validity: which (relation, target_type) combos make sense
    VALID_TARGETS = {
        "father": {"person"},
        "mother": {"person"},
        "wife": {"person"},
        "husband": {"person"},
        "sister": {"person"},
        "brother": {"person"},
        "works_at": {"company"},
        "role": {"person"},  # "software engineer" entity has type person
    }

    # Relations that should only point to non-user, non-work entities
    FAMILY_RELATIONS = {"father", "mother", "wife", "husband", "sister", "brother"}

    def _transe_inference(self, user_entity: Entity, entities_by_id: dict) -> int:
        """
        Use TransE to predict missing links.
        Heavily filtered: only accept predictions that pass semantic validation.
        With <20 triples, TransE is unreliable — we only accept predictions
        that are semantically valid AND score in the top-1 with high margin.
        """
        if not self.model or not self.kg_data:
            return 0

        entity_to_idx = self.kg_data["entity_to_idx"]
        idx_to_entity = self.kg_data["idx_to_entity"]
        idx_to_pred = self.kg_data["idx_to_pred"]
        n_triples = len(self.kg_data["triples"])

        if user_entity.id not in entity_to_idx:
            return 0

        # With very few triples, TransE predictions are unreliable
        # Only use them if we have decent coverage
        if n_triples < 15:
            logger.info(f"  TransE: skipping predictions (only {n_triples} triples, need 15+)")
            return 0

        user_idx = entity_to_idx[user_entity.id]
        predicted = 0

        with torch.no_grad():
            for rel_idx in range(self.model.n_relations):
                rel_name = idx_to_pred[rel_idx]
                predictions = self.model.predict_tails(user_idx, rel_idx, top_k=3)

                valid_types = self.VALID_TARGETS.get(rel_name, set())

                for ent_idx, neg_dist in predictions:
                    if neg_dist > -2.0:  # Strict distance threshold
                        continue
                    if ent_idx == user_idx:
                        continue

                    target_entity = idx_to_entity.get(ent_idx)
                    if not target_entity:
                        continue

                    # Type constraint check
                    if valid_types and target_entity.entity_type not in valid_types:
                        continue

                    # Family relation should not point to companies
                    if rel_name in self.FAMILY_RELATIONS:
                        if target_entity.entity_type == "company":
                            continue
                        # Should not point to "software engineer"
                        if target_entity.name.lower() in ("software engineer", "engineer"):
                            continue

                    # Check if relation already exists
                    existing = self.graph.db.execute(
                        "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                        (user_entity.id, rel_name, target_entity.id)
                    ).fetchone()

                    if not existing:
                        conf = min(0.65, max(0.3, -neg_dist / 3.0))
                        self.graph.add_relation(Relation(
                            subject_id=user_entity.id,
                            predicate=rel_name,
                            object_id=target_entity.id,
                            confidence=conf,
                            is_inferred=True,
                        ))
                        predicted += 1
                        logger.info(f"  TransE predicted: {user_entity.name} --[{rel_name}]--> {target_entity.name}")

        return predicted


# ─── Phase 3 Orchestrator ───────────────────────────────────────

class PatternCompletion:
    """
    Phase 3 orchestrator. Runs:
    1. Entity resolution
    2. Relation mining from memories
    3. TransE training on cleaned graph
    4. Learned inference
    """

    def __init__(self, store, graph, embedding_engine=None,
                 user_name: str = "Dimuthu", model_dir: str = "data/pattern"):
        self.store = store
        self.graph = graph
        self.embeddings = embedding_engine
        self.user_name = user_name
        self.model_dir = model_dir
        self.transe = None
        self.kg_data = None

    def run(self, train_transe_model: bool = True, epochs: int = 500) -> dict:
        """Full Phase 3 pipeline."""
        results = {}
        start = time.time()

        # Step 0: Graph cleanup (garbage entities, contradictions, normalization)
        print("\n  [0/5] Graph Cleanup...")
        cleanup = GraphCleanup(self.graph, self.user_name)
        cleanup_stats = cleanup.cleanup()
        results["cleanup"] = cleanup_stats
        print(f"    Garbage entities removed: {cleanup_stats['garbage_entities_removed']}")
        print(f"    Predicates normalized:    {cleanup_stats['predicates_normalized']}")
        print(f"    Contradictions fixed:     {cleanup_stats['contradictions_fixed']}")
        print(f"    Dangling relations removed: {cleanup_stats['dangling_relations_removed']}")

        # Step 1: Entity resolution
        print("\n  [1/5] Entity Resolution...")
        resolver = EntityResolver(self.graph, self.user_name)
        resolve_stats = resolver.resolve()
        results["entity_resolution"] = resolve_stats
        print(f"    Merged: {resolve_stats['merged']}, "
              f"Reassigned: {resolve_stats['relations_reassigned']}, "
              f"Duplicates: {resolve_stats['duplicates_removed']}, "
              f"Orphans: {resolve_stats['orphans_removed']}")

        # Step 2: Mine relations from memories
        print("\n  [2/5] Mining relations from memories...")
        miner = RelationMiner(self.graph, self.store.db, self.user_name)
        mine_stats = miner.mine_relations()
        results["relation_mining"] = mine_stats
        print(f"    New relations found: {mine_stats['new_relations']}")

        # Step 2b: Re-run cleanup on newly mined relations (fixes spouse direction issues)
        if mine_stats['new_relations'] > 0:
            print("    Post-mining cleanup...")
            post_cleanup = cleanup.cleanup()
            print(f"      Spouse directions fixed: {post_cleanup.get('spouse_direction_fixed', 0)}")
            print(f"      Contradictions fixed:    {post_cleanup.get('contradictions_fixed', 0)}")

        # Step 3: Train TransE (if enough data)
        print("\n  [3/5] Training TransE knowledge graph embeddings...")
        self.kg_data = build_kg_data(self.graph)
        n_triples = len(self.kg_data["triples"])
        print(f"    Entities: {self.kg_data['n_entities']}, "
              f"Relations: {self.kg_data['n_relations']}, "
              f"Triples: {n_triples}")

        transe_stats = {}
        if train_transe_model and n_triples >= 3:
            self.transe = TransE(
                n_entities=self.kg_data["n_entities"],
                n_relations=self.kg_data["n_relations"],
                dim=64,
                margin=1.0,
            )
            total_params = sum(p.numel() for p in self.transe.parameters())
            print(f"    TransE: {total_params:,} parameters")

            history = train_transe(
                self.transe, self.kg_data["triples"],
                self.kg_data["n_entities"], epochs=epochs
            )
            transe_stats = {
                "final_loss": history["loss"][-1],
                "mrr": history["mrr"][-1] if history["mrr"] else 0,
                "params": total_params,
            }

            # Save model
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, "transe.pt")
            torch.save({
                "model_state": self.transe.state_dict(),
                "kg_data": {
                    "entity_to_idx": self.kg_data["entity_to_idx"],
                    "idx_to_pred": self.kg_data["idx_to_pred"],
                    "pred_to_idx": self.kg_data["pred_to_idx"],
                    "n_entities": self.kg_data["n_entities"],
                    "n_relations": self.kg_data["n_relations"],
                },
                "config": {"dim": 64, "margin": 1.0},
            }, model_path)
            print(f"    Model saved to {model_path}")
        else:
            print(f"    Skipping (need >=3 triples, have {n_triples})")

        results["transe"] = transe_stats

        # Step 4: Learned inference
        print("\n  [4/5] Running learned inference...")
        inference = LearnedInference(self.graph, self.transe, self.kg_data)
        infer_stats = inference.run_inference(self.user_name)
        results["inference"] = infer_stats
        print(f"    Rule-based inferred: {infer_stats['inferred']}")
        print(f"    TransE predictions:  {infer_stats['transe_predictions']}")

        results["duration_ms"] = (time.time() - start) * 1000

        # Print final graph state
        print(f"\n  ── Final Graph State ──")
        entities = self.graph.get_all_entities()
        print(f"    Entities: {len(entities)}")
        for e in entities:
            print(f"      {e.name} ({e.entity_type})")

        all_rels = self.graph.get_all_relations(include_inferred=True)
        explicit = [r for r in all_rels if not r.is_inferred]
        inferred = [r for r in all_rels if r.is_inferred]
        entities_by_id = {e.id: e for e in entities}
        print(f"    Explicit relations: {len(explicit)}")
        for r in explicit:
            sn = entities_by_id.get(r.subject_id, Entity(name="?")).name
            on = entities_by_id.get(r.object_id, Entity(name="?")).name
            print(f"      {sn} --[{r.predicate}]--> {on} (conf={r.confidence:.2f})")
        print(f"    Inferred relations: {len(inferred)}")
        for r in inferred:
            sn = entities_by_id.get(r.subject_id, Entity(name="?")).name
            on = entities_by_id.get(r.object_id, Entity(name="?")).name
            print(f"      {sn} --[{r.predicate}]--> {on} (conf={r.confidence:.2f})")

        return results
