"""
Transformer-encoded entity extraction.

Replaces the fragmented spaCy NER → regex → LLM pipeline with a unified
transformer encoder that produces token-level representations used for:
1. Entity span detection (NER via token classification)
2. Entity type classification (person/place/company/animal/concept)
3. Relation predicate classification between entity pairs
4. Node embeddings for graph initialization (same representation space)

The encoder uses the SAME embedding model as the rest of Limbiq
(all-MiniLM-L6-v2 by default), ensuring entity embeddings live in
the same vector space as memory embeddings. This means entity similarity,
memory retrieval, and graph node scoring all share one representation.

Architecture:
    Input text → Transformer encoder → Token embeddings (384-dim)
                                          ├─ Entity span classifier (BIO tagging)
                                          ├─ Entity type classifier (per-span)
                                          ├─ Relation classifier (span-pair)
                                          └─ Node embedding (mean-pooled span)

Falls back gracefully to regex-only extraction if torch unavailable.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import torch for the learned components
_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    pass


# ── Data structures ──────────────────────────────────────────

@dataclass
class EncodedEntity:
    """An entity detected by the transformer encoder."""
    name: str
    entity_type: str              # person, place, company, animal, concept
    confidence: float = 1.0
    embedding: list[float] = field(default_factory=list)  # 384-dim
    span_start: int = 0
    span_end: int = 0


@dataclass
class EncodedRelation:
    """A relation detected between two encoded entities."""
    subject: EncodedEntity
    predicate: str
    object: EncodedEntity
    confidence: float = 1.0


@dataclass
class EncoderOutput:
    """Full output of the transformer encoder pipeline."""
    entities: list[EncodedEntity] = field(default_factory=list)
    relations: list[EncodedRelation] = field(default_factory=list)
    text_embedding: list[float] = field(default_factory=list)  # Full text


# ── Entity type labels ───────────────────────────────────────

ENTITY_TYPES = ["person", "place", "company", "animal", "concept", "condition"]
TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(ENTITY_TYPES)}

# ── Predicate labels (subset of VALID_PREDICATES for classification) ──

PREDICATE_LABELS = [
    "father", "mother", "wife", "husband", "brother", "sister",
    "son", "daughter", "pet", "works_at", "lives_in", "role",
    "friend", "colleague", "boss", "interested_in", "has_condition",
    "related_to",
]
PRED_TO_IDX = {p: i for i, p in enumerate(PREDICATE_LABELS)}
IDX_TO_PRED = {i: p for i, p in enumerate(PREDICATE_LABELS)}


if _torch_available:

    class EntityTypeClassifier(nn.Module):
        """Classifies entity type from mean-pooled span embedding."""

        def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, len(ENTITY_TYPES)),
            )

        def forward(self, span_embedding: torch.Tensor) -> torch.Tensor:
            """Input: (batch, input_dim) → Output: (batch, num_types)"""
            return self.net(span_embedding)

    class RelationClassifier(nn.Module):
        """Classifies relation type from concatenated subject+object embeddings."""

        def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
            super().__init__()
            # Input is [subject_emb; object_emb; element-wise product]
            self.net = nn.Sequential(
                nn.Linear(input_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, len(PREDICATE_LABELS)),
            )

        def forward(self, subj_emb: torch.Tensor, obj_emb: torch.Tensor) -> torch.Tensor:
            """Input: two (batch, dim) tensors → Output: (batch, num_predicates)"""
            combined = torch.cat([subj_emb, obj_emb, subj_emb * obj_emb], dim=-1)
            return self.net(combined)

    class SpanDetector(nn.Module):
        """Detects entity spans from token embeddings using BIO tagging.

        B = Begin entity, I = Inside entity, O = Outside entity
        """

        def __init__(self, input_dim: int = 384, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),  # B, I, O
            )

        def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
            """Input: (seq_len, dim) → Output: (seq_len, 3)"""
            return self.net(token_embeddings)


class TransformerEntityEncoder:
    """Unified transformer encoder for entity extraction and graph embedding.

    Uses the existing EmbeddingEngine's transformer model to produce
    token-level embeddings, then runs lightweight classifiers on top.

    When classifiers are not trained, falls back to heuristic span
    detection (capitalized words, known patterns) but STILL produces
    transformer embeddings for each detected entity — ensuring all
    graph nodes start with semantically meaningful representations.
    """

    def __init__(self, embedding_engine, model_dir: str = "data/encoder"):
        """
        Args:
            embedding_engine: Limbiq's EmbeddingEngine (provides the transformer)
            model_dir: Directory for saving/loading classifier weights
        """
        self.embedding_engine = embedding_engine
        self.model_dir = model_dir
        self._model_loaded = False

        # Lightweight classifiers (only if torch available)
        self.type_classifier = None
        self.relation_classifier = None
        self.span_detector = None

        if _torch_available:
            dim = self._get_embedding_dim()
            self.type_classifier = EntityTypeClassifier(input_dim=dim)
            self.relation_classifier = RelationClassifier(input_dim=dim)
            self.span_detector = SpanDetector(input_dim=dim)
            self._try_load_models()

    def _get_embedding_dim(self) -> int:
        """Detect embedding dimension from the engine."""
        test = self.embedding_engine.embed("test")
        return len(test)

    def _try_load_models(self):
        """Load pretrained classifier weights if available."""
        import os
        type_path = os.path.join(self.model_dir, "type_classifier.pt")
        rel_path = os.path.join(self.model_dir, "relation_classifier.pt")
        span_path = os.path.join(self.model_dir, "span_detector.pt")

        loaded = 0
        for path, model in [
            (type_path, self.type_classifier),
            (rel_path, self.relation_classifier),
            (span_path, self.span_detector),
        ]:
            if os.path.exists(path) and model is not None:
                try:
                    model.load_state_dict(torch.load(path, map_location="cpu"))
                    model.eval()
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if loaded > 0:
            self._model_loaded = True
            logger.info(f"Loaded {loaded}/3 encoder classifier models")

    def encode(self, text: str, user_name: str = "user") -> EncoderOutput:
        """Run the full encoder pipeline on input text.

        1. Tokenize and embed each token via transformer
        2. Detect entity spans (learned or heuristic)
        3. Classify entity types
        4. Classify relations between entity pairs
        5. Produce node embeddings for each entity

        Args:
            text: Input text to extract from
            user_name: User's name (for resolving "my", "I" references)

        Returns:
            EncoderOutput with entities, relations, and embeddings
        """
        if not text or len(text.strip()) < 3:
            return EncoderOutput()

        # Step 1: Get full text embedding
        text_embedding = self.embedding_engine.embed(text)

        # Step 2: Detect entity spans
        # Use learned span detector if available, otherwise heuristic
        if self._model_loaded and self.span_detector is not None:
            spans = self._detect_spans_learned(text)
        else:
            spans = self._detect_spans_heuristic(text)

        if not spans:
            return EncoderOutput(text_embedding=text_embedding)

        # Step 3: Embed each entity span and classify type
        entities = []
        for span_text, start, end in spans:
            # Produce entity embedding via the SAME transformer
            entity_emb = self.embedding_engine.embed(span_text)

            # Classify entity type
            if self._model_loaded and self.type_classifier is not None:
                entity_type = self._classify_type_learned(entity_emb)
            else:
                entity_type = self._classify_type_heuristic(span_text, text)

            entities.append(EncodedEntity(
                name=span_text,
                entity_type=entity_type,
                confidence=0.9 if self._model_loaded else 0.7,
                embedding=entity_emb,
                span_start=start,
                span_end=end,
            ))

        # Step 4: Classify relations between all entity pairs
        relations = []
        if len(entities) >= 2:
            relations = self._classify_relations(entities, text, user_name)

        return EncoderOutput(
            entities=entities,
            relations=relations,
            text_embedding=text_embedding,
        )

    # ── Span detection ───────────────────────────────────────

    def _detect_spans_learned(self, text: str) -> list[tuple[str, int, int]]:
        """Use the trained span detector on token-level embeddings."""
        # Tokenize by whitespace (simple but effective for personal text)
        tokens = text.split()
        if not tokens:
            return []

        # Get per-token embeddings
        token_embs = []
        for token in tokens:
            emb = self.embedding_engine.embed(token)
            token_embs.append(emb)

        token_tensor = torch.tensor(token_embs, dtype=torch.float32)

        with torch.no_grad():
            logits = self.span_detector(token_tensor)  # (seq_len, 3)
            tags = torch.argmax(logits, dim=-1).tolist()  # B=0, I=1, O=2

        # Extract spans from BIO tags
        spans = []
        current_start = None
        current_tokens = []

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag == 0:  # B - Begin
                if current_tokens:
                    span_text = " ".join(current_tokens)
                    char_start = text.find(span_text)
                    spans.append((span_text, char_start, char_start + len(span_text)))
                current_start = i
                current_tokens = [token]
            elif tag == 1 and current_tokens:  # I - Inside
                current_tokens.append(token)
            else:  # O - Outside
                if current_tokens:
                    span_text = " ".join(current_tokens)
                    char_start = text.find(span_text)
                    spans.append((span_text, char_start, char_start + len(span_text)))
                    current_tokens = []

        if current_tokens:
            span_text = " ".join(current_tokens)
            char_start = text.find(span_text)
            spans.append((span_text, char_start, char_start + len(span_text)))

        return [(s, st, en) for s, st, en in spans if len(s) >= 2]

    def _detect_spans_heuristic(self, text: str) -> list[tuple[str, int, int]]:
        """Heuristic span detection — capitalized words and known patterns.

        This is the fallback when classifiers aren't trained. Still
        produces transformer embeddings for each span.
        """
        from limbiq.graph.entities import _is_valid_entity_name

        spans = []
        seen = set()

        # Pattern 1: Capitalized multi-word sequences (proper nouns)
        for m in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
            name = m.group(1).strip()
            if name.lower() not in seen and len(name) >= 2:
                if _is_valid_entity_name(name):
                    seen.add(name.lower())
                    spans.append((name, m.start(), m.end()))

        # Pattern 2: Quoted strings (often entity names)
        for m in re.finditer(r'"([A-Z][a-zA-Z\s]{2,})"', text):
            name = m.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                spans.append((name, m.start(1), m.end(1)))

        return spans

    # ── Type classification ──────────────────────────────────

    def _classify_type_learned(self, entity_emb: list[float]) -> str:
        """Use trained classifier to predict entity type."""
        emb_tensor = torch.tensor([entity_emb], dtype=torch.float32)
        with torch.no_grad():
            logits = self.type_classifier(emb_tensor)
            pred_idx = torch.argmax(logits, dim=-1).item()
        return IDX_TO_TYPE.get(pred_idx, "concept")

    def _classify_type_heuristic(self, entity_name: str, context: str) -> str:
        """Heuristic entity type classification from context clues."""
        name_lower = entity_name.lower()
        ctx_lower = context.lower()

        # Location indicators — but ONLY if no family/person context nearby
        family_context = {"wife", "husband", "mother", "father", "brother",
                          "sister", "son", "daughter", "friend", "colleague",
                          "mom", "dad", "papa", "mama"}
        has_family_context = any(fw in ctx_lower for fw in family_context)
        if not has_family_context:
            if any(f"{w} {name_lower}" in ctx_lower or f"{name_lower} {w}" in ctx_lower
                   for w in ["in", "from", "to"]):
                # Extra check: require an actual location word nearby
                location_words = {"city", "country", "town", "village", "state",
                                  "province", "lives", "located", "based", "visit"}
                if any(lw in ctx_lower for lw in location_words):
                    return "place"

        # Company indicators
        company_suffixes = {"inc", "ltd", "corp", "llc", "co", "company", "group",
                           "technologies", "tech", "labs", "studios"}
        if any(name_lower.endswith(f" {s}") or name_lower == s for s in company_suffixes):
            return "company"
        if "work" in ctx_lower and ("at" in ctx_lower or "for" in ctx_lower):
            # Check proximity
            work_match = re.search(rf'work\w*\s+(?:at|for)\s+{re.escape(entity_name)}', context, re.I)
            if work_match:
                return "company"

        # Animal indicators
        animal_words = {"dog", "cat", "pet", "puppy", "kitten", "animal"}
        if any(w in ctx_lower for w in animal_words):
            pet_match = re.search(
                rf'(?:my|our)\s+(?:dog|cat|pet)\s+\w*\s*{re.escape(entity_name)}',
                context, re.I
            )
            if pet_match:
                return "animal"

        # Condition indicators
        condition_words = {"condition", "disease", "syndrome", "disorder",
                          "diagnosed", "has"}
        if any(w in ctx_lower for w in condition_words):
            cond_match = re.search(
                rf'{re.escape(entity_name)}\s+(?:condition|disease|syndrome)',
                context, re.I
            )
            if cond_match:
                return "condition"

        # Default to person (most common in personal knowledge graphs)
        return "person"

    # ── Relation classification ──────────────────────────────

    def _classify_relations(
        self, entities: list[EncodedEntity], text: str, user_name: str
    ) -> list[EncodedRelation]:
        """Classify relations between entity pairs.

        Uses learned classifier when available, otherwise context-based heuristics.
        Only considers pairs that appear in the same sentence or near each other.
        """
        relations = []

        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue

                # Check proximity (within ~100 chars of each other in text)
                distance = abs(subj.span_start - obj.span_start)
                if distance > 150:
                    continue

                if self._model_loaded and self.relation_classifier is not None:
                    pred, conf = self._classify_relation_learned(subj, obj)
                else:
                    pred, conf = self._classify_relation_heuristic(
                        subj, obj, text, user_name
                    )

                if pred and conf > 0.3:
                    relations.append(EncodedRelation(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=conf,
                    ))

        return relations

    def _classify_relation_learned(
        self, subj: EncodedEntity, obj: EncodedEntity
    ) -> tuple[Optional[str], float]:
        """Use trained relation classifier."""
        subj_t = torch.tensor([subj.embedding], dtype=torch.float32)
        obj_t = torch.tensor([obj.embedding], dtype=torch.float32)

        with torch.no_grad():
            logits = self.relation_classifier(subj_t, obj_t)
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred_idx].item()

        pred = IDX_TO_PRED.get(pred_idx)
        return pred, conf

    def _classify_relation_heuristic(
        self, subj: EncodedEntity, obj: EncodedEntity,
        text: str, user_name: str
    ) -> tuple[Optional[str], float]:
        """Context-based relation classification.

        IMPORTANT: Only creates a relation when the keyword appears
        BETWEEN the two entity mentions (not anywhere in the text).
        This prevents "Prabhashi's father is Chandrasiri" from also
        creating "Renuka→father→Chandrasiri" just because "father"
        appears somewhere in the text.
        """
        # Extract text strictly between the two entity spans
        first_end = min(subj.span_end, obj.span_end)
        second_start = max(subj.span_start, obj.span_start)
        if first_end >= second_start:
            # Overlapping or adjacent — no space for relation keyword
            return None, 0.0
        between = text[first_end:second_start].lower().strip()

        if not between:
            return None, 0.0

        # Stop at clause boundaries — don't match across commas, periods,
        # semicolons, or "and"/"but" conjunctions. This prevents
        # "Prabhashi's father is Chandrasiri, My Dog is Dexter" from
        # creating a relation between Prabhashi and Dexter via "father".
        clause_break = re.search(r'[,;.]|\band\b|\bbut\b', between)
        if clause_break:
            # Only use text up to the first clause break
            between = between[:clause_break.start()].strip()

        # Determine which entity came first in the text
        if subj.span_start < obj.span_start:
            first_ent, second_ent = subj, obj
        else:
            first_ent, second_ent = obj, subj

        # Family relations — keyword must be BETWEEN the entities
        # Patterns: "X's father is Y" → X→father→Y
        #           "X father Y"      → X→father→Y
        #
        # IMPORTANT: Skip when keyword is preceded by "my"/"our" — that
        # indicates the USER's relation, not a relation between the two
        # entities. E.g., "Prabhashi called my mom Renuka" → "my mom" is
        # the user's mother, NOT Prabhashi's mother.
        family_map = {
            "father": ["father", "dad", "papa", "pa"],
            "mother": ["mother", "mom", "mum", "mama", "ma"],
            "wife": ["wife", "married to", "spouse"],
            "husband": ["husband", "married to", "spouse"],
            "brother": ["brother", "bro"],
            "sister": ["sister", "sis"],
            "son": ["son"],
            "daughter": ["daughter"],
            "pet": ["dog", "cat", "pet", "puppy", "animal"],
        }

        for pred, keywords in family_map.items():
            for kw in keywords:
                if kw in between:
                    # If keyword is preceded by "my" or "our", this is the
                    # USER's relation, not a relation between these entities.
                    # E.g., "called my mom" → user's mother, skip.
                    if re.search(rf'\b(?:my|our)\s+{re.escape(kw)}\b', between):
                        continue
                    if first_ent is subj:
                        return pred, 0.7
                    else:
                        return pred, 0.7

        # Work/location relations
        if "work" in between and ("at" in between or "for" in between):
            return "works_at", 0.7
        if "live" in between and "in" in between:
            return "lives_in", 0.7

        return None, 0.0

    # ── Training ─────────────────────────────────────────────

    def train_from_graph(self, graph_store, num_epochs: int = 50) -> dict:
        """Train classifiers from existing graph data (self-supervised).

        Uses the current graph as ground truth:
        - Entity names + types → type classifier training pairs
        - Relations → relation classifier training pairs
        - Memory mentions → span detector training data

        This is a bootstrap training — the graph was built by regex/LLM,
        and now the transformer classifiers learn to replicate it.
        """
        if not _torch_available:
            return {"status": "torch_unavailable"}

        import os
        os.makedirs(self.model_dir, exist_ok=True)

        entities = graph_store.get_all_entities()
        relations = graph_store.get_all_relations(include_inferred=False)

        if len(entities) < 3:
            return {"status": "insufficient_data", "entities": len(entities)}

        # ── Train type classifier ────────────────────────────
        type_X = []
        type_y = []
        for ent in entities:
            if ent.entity_type in TYPE_TO_IDX:
                emb = self.embedding_engine.embed(ent.name)
                type_X.append(emb)
                type_y.append(TYPE_TO_IDX[ent.entity_type])

        type_loss = 0.0
        if len(type_X) >= 3:
            type_X_t = torch.tensor(type_X, dtype=torch.float32)
            type_y_t = torch.tensor(type_y, dtype=torch.long)

            self.type_classifier.train()
            optimizer = torch.optim.Adam(self.type_classifier.parameters(), lr=1e-3)

            for epoch in range(num_epochs):
                logits = self.type_classifier(type_X_t)
                loss = F.cross_entropy(logits, type_y_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                type_loss = loss.item()

            self.type_classifier.eval()
            torch.save(
                self.type_classifier.state_dict(),
                os.path.join(self.model_dir, "type_classifier.pt"),
            )

        # ── Train relation classifier ────────────────────────
        rel_X_subj = []
        rel_X_obj = []
        rel_y = []
        entity_map = {e.id: e for e in entities}

        for rel in relations:
            if rel.predicate not in PRED_TO_IDX:
                continue
            subj = entity_map.get(rel.subject_id)
            obj = entity_map.get(rel.object_id)
            if subj and obj:
                subj_emb = self.embedding_engine.embed(subj.name)
                obj_emb = self.embedding_engine.embed(obj.name)
                rel_X_subj.append(subj_emb)
                rel_X_obj.append(obj_emb)
                rel_y.append(PRED_TO_IDX[rel.predicate])

        rel_loss = 0.0
        if len(rel_y) >= 3:
            subj_t = torch.tensor(rel_X_subj, dtype=torch.float32)
            obj_t = torch.tensor(rel_X_obj, dtype=torch.float32)
            rel_y_t = torch.tensor(rel_y, dtype=torch.long)

            self.relation_classifier.train()
            optimizer = torch.optim.Adam(self.relation_classifier.parameters(), lr=1e-3)

            for epoch in range(num_epochs):
                logits = self.relation_classifier(subj_t, obj_t)
                loss = F.cross_entropy(logits, rel_y_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rel_loss = loss.item()

            self.relation_classifier.eval()
            torch.save(
                self.relation_classifier.state_dict(),
                os.path.join(self.model_dir, "relation_classifier.pt"),
            )

        self._model_loaded = True
        return {
            "status": "trained",
            "type_samples": len(type_X),
            "relation_samples": len(rel_y),
            "type_loss": round(type_loss, 4),
            "relation_loss": round(rel_loss, 4),
        }
