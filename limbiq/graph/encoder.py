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
    "related_to", "none",
]
PRED_TO_IDX = {p: i for i, p in enumerate(PREDICATE_LABELS)}
IDX_TO_PRED = {i: p for i, p in enumerate(PREDICATE_LABELS)}

# Model version — incremented when architecture changes (skip loading old weights)
_RELATION_CLASSIFIER_VERSION = 2


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

    class ContextualRelationClassifier(nn.Module):
        """Self-attention relation classifier that sees sentence context.

        Unlike the old MLP that only saw [subj_emb; obj_emb; product],
        this module attends over the full sentence tokens — especially the
        words BETWEEN entities — to determine directionality and predicate.

        Architecture (~55K params):
          1. Project tokens: 384-dim → 64-dim
          2. Add learned entity markers to subject/object spans
          3. Single-layer self-attention (2 heads, 64-dim)
          4. Entity-aware pooling: [subj_repr; between_repr; obj_repr]
          5. Classify: 192-dim → 19 predicates (18 + "none")
        """

        def __init__(self, input_dim: int = 384, proj_dim: int = 64):
            super().__init__()
            self.proj_dim = proj_dim

            # Step 1: Project down
            self.projection = nn.Linear(input_dim, proj_dim)

            # Step 2: Learned entity markers
            self.subj_marker = nn.Parameter(torch.randn(proj_dim) * 0.02)
            self.obj_marker = nn.Parameter(torch.randn(proj_dim) * 0.02)

            # Step 3: Self-attention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=proj_dim, num_heads=2, batch_first=True,
                dropout=0.1,
            )
            self.attn_norm = nn.LayerNorm(proj_dim)
            self.ff = nn.Sequential(
                nn.Linear(proj_dim, proj_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(proj_dim * 2, proj_dim),
            )
            self.ff_norm = nn.LayerNorm(proj_dim)

            # Step 5: Classification head
            # Input: [subj_repr; between_repr; obj_repr] = 3 * proj_dim
            self.classifier = nn.Sequential(
                nn.Linear(proj_dim * 3, proj_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(proj_dim, len(PREDICATE_LABELS)),
            )

        def forward(
            self,
            token_embs: torch.Tensor,
            subj_mask: torch.Tensor,
            obj_mask: torch.Tensor,
            between_mask: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                token_embs: (batch, seq_len, input_dim) token embeddings
                subj_mask: (batch, seq_len) 1.0 for subject tokens
                obj_mask: (batch, seq_len) 1.0 for object tokens
                between_mask: (batch, seq_len) 1.0 for tokens between entities

            Returns:
                (batch, num_predicates) logits
            """
            # Step 1: Project down
            x = self.projection(token_embs)  # (batch, seq_len, proj_dim)

            # Step 2: Inject entity markers
            x = x + subj_mask.unsqueeze(-1) * self.subj_marker
            x = x + obj_mask.unsqueeze(-1) * self.obj_marker

            # Step 3: Self-attention + residual + norm
            attn_out, _ = self.self_attn(x, x, x)
            x = self.attn_norm(x + attn_out)

            # Feed-forward + residual + norm
            ff_out = self.ff(x)
            x = self.ff_norm(x + ff_out)

            # Step 4: Entity-aware pooling
            subj_repr = self._masked_mean(x, subj_mask)
            obj_repr = self._masked_mean(x, obj_mask)
            between_repr = self._masked_mean(x, between_mask)

            # Step 5: Classify
            combined = torch.cat([subj_repr, between_repr, obj_repr], dim=-1)
            return self.classifier(combined)

        @staticmethod
        def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Mean pool over masked positions. Falls back to zero if no positions."""
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq, 1)
            count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
            return (x * mask_expanded).sum(dim=1) / count  # (batch, dim)

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
            self.relation_classifier = ContextualRelationClassifier(input_dim=dim)
            self.span_detector = SpanDetector(input_dim=dim)
            self._try_load_models()

    def _get_embedding_dim(self) -> int:
        """Detect embedding dimension from the engine."""
        test = self.embedding_engine.embed("test")
        return len(test)

    def _try_load_models(self):
        """Load pretrained classifier weights if available.

        Uses version check for relation classifier — old MLP weights
        are incompatible with the new ContextualRelationClassifier.
        """
        import os
        type_path = os.path.join(self.model_dir, "type_classifier.pt")
        rel_path = os.path.join(self.model_dir, "relation_classifier_v2.pt")
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
        """Classify relations between entity pairs using self-attention.

        The contextual classifier sees the full sentence tokens and attends
        to words between entities to determine directionality and predicate.
        Falls back gracefully when torch/classifier not available.
        """
        relations = []

        if not (_torch_available and self.relation_classifier is not None):
            return relations  # Defer to regex pipeline in entities.py

        # Get token-level embeddings for the sentence
        tokens = text.split()
        if not tokens:
            return relations

        token_embs = [self.embedding_engine.embed(t) for t in tokens]
        token_tensor = torch.tensor([token_embs], dtype=torch.float32)  # (1, seq, dim)
        seq_len = len(tokens)

        # Build character-to-token index for span mapping
        char_to_token = {}
        pos = 0
        for i, token in enumerate(tokens):
            start = text.find(token, pos)
            for c in range(start, start + len(token)):
                char_to_token[c] = i
            pos = start + len(token)

        for i, subj in enumerate(entities):
            for j, obj in enumerate(entities):
                if i == j:
                    continue

                distance = abs(subj.span_start - obj.span_start)
                if distance > 150:
                    continue

                # Build masks
                subj_mask = torch.zeros(1, seq_len)
                obj_mask = torch.zeros(1, seq_len)
                between_mask = torch.zeros(1, seq_len)

                for c in range(subj.span_start, subj.span_end):
                    if c in char_to_token:
                        subj_mask[0, char_to_token[c]] = 1.0

                for c in range(obj.span_start, obj.span_end):
                    if c in char_to_token:
                        obj_mask[0, char_to_token[c]] = 1.0

                # Between = tokens between the two entity spans
                between_start = min(subj.span_end, obj.span_end)
                between_end = max(subj.span_start, obj.span_start)
                for c in range(between_start, between_end):
                    if c in char_to_token:
                        between_mask[0, char_to_token[c]] = 1.0

                # If no between tokens, use a small window around entities
                if between_mask.sum() == 0:
                    continue  # Adjacent/overlapping entities, skip

                pred, conf = self._classify_relation_contextual(
                    token_tensor, subj_mask, obj_mask, between_mask
                )

                if pred and pred != "none" and conf > 0.3:
                    relations.append(EncodedRelation(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=conf,
                    ))

        return relations

    def _classify_relation_contextual(
        self,
        token_embs: "torch.Tensor",
        subj_mask: "torch.Tensor",
        obj_mask: "torch.Tensor",
        between_mask: "torch.Tensor",
    ) -> tuple[Optional[str], float]:
        """Use the contextual self-attention classifier."""
        with torch.no_grad():
            logits = self.relation_classifier(
                token_embs, subj_mask, obj_mask, between_mask
            )
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred_idx].item()

        pred = IDX_TO_PRED.get(pred_idx)
        return pred, conf

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

        # ── Train contextual relation classifier ────────────
        rel_loss = 0.0
        rel_samples = self._train_relation_classifier(
            graph_store, entities, relations, num_epochs
        )
        if isinstance(rel_samples, dict):
            rel_loss = rel_samples.get("loss", 0.0)
            rel_count = rel_samples.get("samples", 0)
        else:
            rel_count = 0

        self._model_loaded = True
        return {
            "status": "trained",
            "type_samples": len(type_X),
            "relation_samples": rel_count,
            "type_loss": round(type_loss, 4),
            "relation_loss": round(rel_loss, 4),
        }

    def _train_relation_classifier(
        self, graph_store, entities, relations, num_epochs: int = 50
    ) -> dict:
        """Train the contextual relation classifier using synthetic sentences.

        Generates training data by synthesizing canonical sentences from
        graph relations and optional user corrections. The self-attention
        module needs token-level input with entity span masks.
        """
        import os
        if not _torch_available or self.relation_classifier is None:
            return {"samples": 0, "loss": 0.0}

        entity_map = {e.id: e for e in entities}

        # ── Sentence templates for bootstrap training ──
        TEMPLATES = {
            "father": ["{subj}'s father is {obj}", "{obj} is {subj}'s father"],
            "mother": ["{subj}'s mother is {obj}", "{obj} is {subj}'s mother"],
            "wife": ["{subj}'s wife is {obj}", "{obj} is {subj}'s wife"],
            "husband": ["{subj}'s husband is {obj}", "{obj} is {subj}'s husband"],
            "brother": ["{subj}'s brother is {obj}", "{obj} is {subj}'s brother"],
            "sister": ["{subj}'s sister is {obj}", "{obj} is {subj}'s sister"],
            "son": ["{subj}'s son is {obj}", "{obj} is {subj}'s son"],
            "daughter": ["{subj}'s daughter is {obj}", "{obj} is {subj}'s daughter"],
            "pet": ["{subj}'s pet is {obj}", "{subj} has a pet named {obj}"],
            "works_at": ["{subj} works at {obj}", "{subj} is employed at {obj}"],
            "lives_in": ["{subj} lives in {obj}", "{subj} is based in {obj}"],
            "friend": ["{subj}'s friend is {obj}", "{obj} is {subj}'s friend"],
            "colleague": ["{subj}'s colleague is {obj}", "{obj} is {subj}'s colleague"],
        }

        training_data = []  # [(sentence, subj_name, obj_name, predicate_idx, weight)]

        # Generate from graph relations
        for rel in relations:
            if rel.predicate not in PRED_TO_IDX:
                continue
            subj = entity_map.get(rel.subject_id)
            obj = entity_map.get(rel.object_id)
            if not subj or not obj:
                continue

            templates = TEMPLATES.get(rel.predicate, ["{subj} {obj}"])
            for template in templates:
                sentence = template.format(subj=subj.name, obj=obj.name)
                training_data.append((
                    sentence, subj.name, obj.name,
                    PRED_TO_IDX[rel.predicate], 1.0
                ))

        # Add user corrections (5x weight)
        try:
            corrections = graph_store.get_relation_corrections()
            for corr in corrections:
                if corr["predicate"] not in PRED_TO_IDX:
                    continue
                target = PRED_TO_IDX[corr["predicate"]] if corr["is_positive"] \
                    else PRED_TO_IDX["none"]
                weight = 5.0
                training_data.append((
                    corr["sentence"], corr["subject_name"], corr["object_name"],
                    target, weight
                ))
        except Exception:
            pass  # No corrections table yet

        if len(training_data) < 3:
            return {"samples": 0, "loss": 0.0}

        # Build training tensors
        all_token_embs = []
        all_subj_masks = []
        all_obj_masks = []
        all_between_masks = []
        all_labels = []
        all_weights = []

        for sentence, subj_name, obj_name, label, weight in training_data:
            tokens = sentence.split()
            if not tokens:
                continue

            # Embed tokens
            token_embs = [self.embedding_engine.embed(t) for t in tokens]

            # Find subject and object token positions
            subj_mask = [0.0] * len(tokens)
            obj_mask = [0.0] * len(tokens)
            between_mask = [0.0] * len(tokens)

            subj_start = subj_end = obj_start = obj_end = -1
            subj_tokens = subj_name.split()
            obj_tokens = obj_name.split()

            # Find subject span
            for k in range(len(tokens) - len(subj_tokens) + 1):
                if tokens[k:k + len(subj_tokens)] == subj_tokens:
                    subj_start = k
                    subj_end = k + len(subj_tokens)
                    for m in range(subj_start, subj_end):
                        subj_mask[m] = 1.0
                    break

            # Find object span
            for k in range(len(tokens) - len(obj_tokens) + 1):
                if tokens[k:k + len(obj_tokens)] == obj_tokens:
                    obj_start = k
                    obj_end = k + len(obj_tokens)
                    for m in range(obj_start, obj_end):
                        obj_mask[m] = 1.0
                    break

            if subj_start == -1 or obj_start == -1:
                continue

            # Between mask
            btwn_start = min(subj_end, obj_end)
            btwn_end = max(subj_start, obj_start)
            for k in range(btwn_start, btwn_end):
                between_mask[k] = 1.0

            all_token_embs.append(token_embs)
            all_subj_masks.append(subj_mask)
            all_obj_masks.append(obj_mask)
            all_between_masks.append(between_mask)
            all_labels.append(label)
            all_weights.append(weight)

        if not all_labels:
            return {"samples": 0, "loss": 0.0}

        # Pad sequences to same length
        max_len = max(len(t) for t in all_token_embs)
        dim = len(all_token_embs[0][0])

        padded_embs = []
        padded_subj = []
        padded_obj = []
        padded_btwn = []
        for embs, sm, om, bm in zip(
            all_token_embs, all_subj_masks, all_obj_masks, all_between_masks
        ):
            pad_len = max_len - len(embs)
            padded_embs.append(embs + [[0.0] * dim] * pad_len)
            padded_subj.append(sm + [0.0] * pad_len)
            padded_obj.append(om + [0.0] * pad_len)
            padded_btwn.append(bm + [0.0] * pad_len)

        embs_t = torch.tensor(padded_embs, dtype=torch.float32)
        subj_t = torch.tensor(padded_subj, dtype=torch.float32)
        obj_t = torch.tensor(padded_obj, dtype=torch.float32)
        btwn_t = torch.tensor(padded_btwn, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.long)
        weights_t = torch.tensor(all_weights, dtype=torch.float32)

        # Train
        self.relation_classifier.train()
        optimizer = torch.optim.AdamW(
            self.relation_classifier.parameters(), lr=1e-3, weight_decay=1e-2
        )

        rel_loss = 0.0
        for epoch in range(num_epochs):
            logits = self.relation_classifier(embs_t, subj_t, obj_t, btwn_t)
            # Weighted cross-entropy
            per_sample_loss = F.cross_entropy(logits, labels_t, reduction="none")
            loss = (per_sample_loss * weights_t).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rel_loss = loss.item()

        self.relation_classifier.eval()
        torch.save(
            self.relation_classifier.state_dict(),
            os.path.join(self.model_dir, "relation_classifier_v2.pt"),
        )

        return {"samples": len(all_labels), "loss": round(rel_loss, 4)}

    def incremental_train(self, graph_store, num_epochs: int = 20) -> dict:
        """Fine-tune relation classifier using correction feedback.

        Called when enough user corrections have accumulated (>= 3).
        Combines corrections (5x weight) with graph relations (1x weight).
        """
        if not _torch_available:
            return {"status": "torch_unavailable"}

        import os
        os.makedirs(self.model_dir, exist_ok=True)

        entities = graph_store.get_all_entities()
        relations = graph_store.get_all_relations(include_inferred=False)

        result = self._train_relation_classifier(
            graph_store, entities, relations, num_epochs
        )

        self._model_loaded = True
        return {
            "status": "retrained",
            "relation_samples": result.get("samples", 0),
            "relation_loss": result.get("loss", 0.0),
        }
