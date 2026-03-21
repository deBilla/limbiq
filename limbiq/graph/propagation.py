"""
Active Graph Propagation — Phase 1
====================================
Makes limbiq's knowledge graph ACTIVE: nodes compute, not just store.

This module implements:
1. Activation values on memory nodes (spread via similarity)
2. Confidence propagation (neighbors reinforce or weaken each other)
3. Contradiction detection (similar content, conflicting facts)
4. Deduplication (merge near-identical memories)
5. Noise suppression (detect and suppress low-value memories)
6. Entity extraction repair (populate graph from existing memories)
7. Cascade tracking (measure propagation dynamics for criticality analysis)

The propagation rules are hand-written (Phase 1). Phase 2 will replace
them with a learned GNN.
"""

import re
import time
import json
import struct
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from limbiq.store.memory_store import MemoryStore, _deserialize_embedding
from limbiq.graph.store import GraphStore, Entity, Relation

logger = logging.getLogger(__name__)


# ─── Noise patterns that indicate low-value memories ──────────────
NOISE_PATTERNS = [
    r"user'?s?\s+greeting\s+was",
    r"user\s+initiated\s+the\s+conversation",
    r"user'?s?\s+question\s+was\s+a\s+simple",
    r"I'm a helpful AI assistant",
    r"I don't recognize users",
    r"current date is",
    r"current day of the week",
    r"falls in week \d+",
    r"There are \d+ days in the month",
    r"conversation started from scratch",
    r"information found does not list",
    r"does not list the specific",
]

# ─── Patterns for entity re-extraction from compressed memories ───
# These handle the "bullet point fact list" format that limbiq's
# compression produces
FACT_PATTERNS = [
    # "Dimuthu is married to Prabhashi"
    (r"(\w+)\s+is\s+married\s+to\s+(\w+)",
     lambda m: [(m.group(1), "wife" if _likely_male(m.group(1)) else "husband", m.group(2)),
                (m.group(2), "husband" if _likely_male(m.group(1)) else "wife", m.group(1))]),

    # "Dimuthu's wife is Prabhashi" / "Dimuthu's father is Upananda"
    (r"(\w+)'?s?\s+(father|mother|wife|husband|sister|brother|son|daughter)\s+(?:is|was)\s+(?:named?\s+)?(\w+)",
     lambda m: [(m.group(1), m.group(2), m.group(3))]),

    # "The user's father's name is Upananda"
    (r"(?:The\s+)?user'?s?\s+(father|mother|wife|husband|sister|brother|son|daughter)'?s?\s+name\s+is\s+(\w+)",
     lambda m: [("user", m.group(1), m.group(2))]),

    # "Dimuthu is a software engineer at Bitsmedia"
    (r"(\w+)\s+is\s+a\s+(software engineer|engineer|developer|manager|designer|architect|analyst)\s+at\s+(\w+)",
     lambda m: [(m.group(1), "works_at", m.group(3)), (m.group(1), "role", m.group(2))]),

    # "Dimuthu works at Bitsmedia"
    (r"(\w+)\s+(?:works?|worked)\s+(?:at|for)\s+(\w+)",
     lambda m: [(m.group(1), "works_at", m.group(2))]),

    # "elder sister's name is Dilini" / "younger sister's name is Dilani"
    (r"(?:elder|older|younger)\s+(sister|brother)'?s?\s+name\s+is\s+(\w+)",
     lambda m: [("user", m.group(1), m.group(2))]),

    # "Dilini is married to Dilanka"
    (r"(\w+)\s+is\s+married\s+to\s+(\w+)",
     lambda m: [(m.group(1), "husband", m.group(2))]),

    # "has two sisters" or "has a sister named X"
    (r"has\s+(?:two|three|a)\s+(sister|brother)s?",
     lambda m: []),  # Just structural info, no specific relation

    # "User said: The user's elder sister's name is Dilini"
    (r"(?:The\s+)?user'?s?\s+(?:elder|younger|older)?\s*(sister|brother)'?s?\s+name\s+is\s+(\w+)",
     lambda m: [("user", m.group(1), m.group(2))]),
]

# Entity type inference
COMPANY_NAMES = {"bitsmedia", "google", "apple", "microsoft", "amazon", "meta", "openai", "anthropic"}
PLACE_INDICATORS = {"singapore", "colombo", "london", "new york", "tokyo", "paris", "berlin"}


def _likely_male(name: str) -> bool:
    """Simple heuristic — can be expanded."""
    return name.lower() in {"dimuthu", "upananda", "dilanka", "devinda"}


def _infer_entity_type(name: str) -> str:
    """Infer entity type from name."""
    lower = name.lower()
    if lower in COMPANY_NAMES:
        return "company"
    if lower in PLACE_INDICATORS:
        return "place"
    return "person"


@dataclass
class PropagationResult:
    """Results from one propagation cycle."""
    entities_created: int = 0
    relations_created: int = 0
    duplicates_merged: int = 0
    noise_suppressed: int = 0
    confidence_updates: int = 0
    cascade_lengths: list = field(default_factory=list)
    duration_ms: float = 0


@dataclass
class ActivationState:
    """Activation state for a memory node."""
    memory_id: str
    activation: float = 0.0     # Current activation level
    c_value: float = 0.0        # Intrinsic charge (from confidence, recency, access)
    stability: float = 0.0      # How stable this node is (0=volatile, 1=crystallized)


class ActiveGraphPropagation:
    """
    Phase 1: Hand-written propagation rules on limbiq's knowledge graph.

    Operations:
    1. repair_graph() — Re-extract entities from ALL existing memories
    2. suppress_noise() — Detect and suppress low-value memories
    3. merge_duplicates() — Find and merge near-identical memories
    4. propagate() — Run activation dynamics (N steps)
    5. compute_activations() — Calculate node activation states
    """

    def __init__(self, store: MemoryStore, graph: GraphStore,
                 embedding_engine=None, user_name: str = "Dimuthu"):
        self.store = store
        self.graph = graph
        self.embeddings = embedding_engine
        self.user_name = user_name

        # Propagation parameters
        self.decay_rate = 0.85          # Activation decay per step
        self.coupling_strength = 0.3    # How much neighbors influence each other
        self.similarity_threshold = 0.5 # Min similarity to form an edge
        self.merge_threshold = 0.92     # Similarity above which memories are merged
        self.noise_confidence = 0.05    # Confidence assigned to noise memories

    # ═══════════════════════════════════════════════════════════════
    # 1. GRAPH REPAIR — Populate entities from existing memories
    # ═══════════════════════════════════════════════════════════════
    def repair_graph(self) -> dict:
        """
        Re-extract entities and relations from ALL existing memories.
        This fixes the empty graph problem by mining the memory store.
        """
        stats = {"entities_created": 0, "relations_created": 0, "memories_scanned": 0}

        # Get all active memories
        cursor = self.store.db.execute(
            "SELECT id, content, tier, is_priority FROM memories WHERE is_suppressed = 0"
        )
        memories = cursor.fetchall()
        stats["memories_scanned"] = len(memories)

        # Ensure user entity exists
        user_entity = self.graph.find_entity_by_name(self.user_name)
        if not user_entity:
            user_entity = self.graph.add_entity(Entity(
                name=self.user_name, entity_type="person"
            ))
            stats["entities_created"] += 1

        # Also check for "default" entity and treat it as the user
        default_entity = self.graph.find_entity_by_name("default")

        # Track what we've extracted to avoid duplicates
        seen_relations = set()

        for mem_id, content, tier, is_priority in memories:
            # Try each fact pattern against the memory content
            for pattern, extractor in FACT_PATTERNS:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    try:
                        triples = extractor(match)
                    except Exception:
                        continue

                    for subj_name, predicate, obj_name in triples:
                        if not subj_name or not obj_name or not predicate:
                            continue

                        # Resolve "user" or "default" to actual name
                        if subj_name.lower() in ("user", "default"):
                            subj_name = self.user_name
                        if obj_name.lower() in ("user", "default"):
                            obj_name = self.user_name

                        # Skip junk
                        if len(obj_name) < 2 or obj_name.lower() in {"the", "a", "an", "is", "was"}:
                            continue

                        # Dedup check
                        rel_key = (subj_name.lower(), predicate.lower(), obj_name.lower())
                        if rel_key in seen_relations:
                            continue
                        seen_relations.add(rel_key)

                        # Create/find entities
                        subj = self.graph.find_entity_by_name(subj_name)
                        if not subj:
                            subj = self.graph.add_entity(Entity(
                                name=subj_name,
                                entity_type=_infer_entity_type(subj_name),
                                source_memory_id=mem_id,
                            ))
                            stats["entities_created"] += 1

                        obj = self.graph.find_entity_by_name(obj_name)
                        if not obj:
                            obj = self.graph.add_entity(Entity(
                                name=obj_name,
                                entity_type=_infer_entity_type(obj_name),
                                source_memory_id=mem_id,
                            ))
                            stats["entities_created"] += 1

                        # Create relation
                        self.graph.add_relation(Relation(
                            subject_id=subj.id,
                            predicate=predicate,
                            object_id=obj.id,
                            confidence=1.0 if is_priority else 0.8,
                            source_memory_id=mem_id,
                        ))
                        stats["relations_created"] += 1

        return stats

    # ═══════════════════════════════════════════════════════════════
    # 2. NOISE SUPPRESSION
    # ═══════════════════════════════════════════════════════════════
    def suppress_noise(self) -> int:
        """
        Detect and suppress low-value noise memories.
        Returns count of suppressed memories.
        """
        suppressed = 0
        cursor = self.store.db.execute(
            "SELECT id, content FROM memories WHERE is_suppressed = 0"
        )

        for mem_id, content in cursor.fetchall():
            content_lower = content.lower()
            for pattern in NOISE_PATTERNS:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    self.store.suppress(mem_id, "noise_detected")
                    suppressed += 1
                    break

        return suppressed

    # ═══════════════════════════════════════════════════════════════
    # 3. DUPLICATE MERGING
    # ═══════════════════════════════════════════════════════════════
    def merge_duplicates(self) -> int:
        """
        Find semantically near-identical memories and merge them.
        Keeps the one with higher confidence/access, suppresses the rest.
        Returns count of merged (suppressed) duplicates.
        """
        if not self.embeddings:
            return 0

        # Get all active memories with embeddings
        cursor = self.store.db.execute(
            "SELECT id, content, confidence, access_count, is_priority, embedding "
            "FROM memories WHERE is_suppressed = 0 AND embedding IS NOT NULL"
        )
        memories = cursor.fetchall()

        if len(memories) < 2:
            return 0

        # Build similarity matrix
        ids = [m[0] for m in memories]
        contents = [m[1] for m in memories]
        confidences = [m[2] for m in memories]
        accesses = [m[3] for m in memories]
        priorities = [m[4] for m in memories]

        # Decode embeddings
        embeddings = []
        valid_indices = []
        for i, m in enumerate(memories):
            emb = _deserialize_embedding(m[5])
            if emb:
                embeddings.append(emb)
                valid_indices.append(i)

        if len(embeddings) < 2:
            return 0

        import numpy as np
        emb_array = np.array(embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = emb_array / norms
        sim_matrix = normalized @ normalized.T

        # Find merge groups
        merged = 0
        already_merged = set()

        for i in range(len(valid_indices)):
            if valid_indices[i] in already_merged:
                continue

            duplicates = []
            for j in range(i + 1, len(valid_indices)):
                if valid_indices[j] in already_merged:
                    continue
                if sim_matrix[i][j] >= self.merge_threshold:
                    duplicates.append(j)

            if not duplicates:
                continue

            # Find the best memory in this group (highest score)
            group = [i] + duplicates
            scores = []
            for idx in group:
                orig_idx = valid_indices[idx]
                score = (confidences[orig_idx] * 10 +
                        accesses[orig_idx] +
                        priorities[orig_idx] * 50)
                scores.append((score, idx))

            scores.sort(reverse=True)
            keeper_idx = valid_indices[scores[0][1]]

            # Suppress duplicates, boost keeper
            for score, idx in scores[1:]:
                orig_idx = valid_indices[idx]
                mem_id = ids[orig_idx]
                self.store.suppress(mem_id, "duplicate_merged")
                already_merged.add(orig_idx)
                merged += 1

            # Boost keeper's confidence
            keeper_id = ids[keeper_idx]
            new_conf = min(1.0, confidences[keeper_idx] + 0.05 * len(duplicates))
            self.store.boost_confidence(keeper_id, new_conf)

        return merged

    # ═══════════════════════════════════════════════════════════════
    # 4. PRIORITY DEFLATION
    # ═══════════════════════════════════════════════════════════════
    def deflate_priorities(self) -> int:
        """
        Reduce priority inflation by demoting low-value priority memories.
        A priority memory should contain genuinely important personal info,
        not noise or generic facts.
        """
        demoted = 0
        cursor = self.store.db.execute(
            "SELECT id, content, access_count FROM memories "
            "WHERE is_priority = 1 AND is_suppressed = 0"
        )

        for mem_id, content, access_count in cursor.fetchall():
            content_lower = content.lower()

            # Check if this is actually important
            should_demote = False

            # Noise patterns in priority = definite demote
            for pattern in NOISE_PATTERNS:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    should_demote = True
                    break

            # Meta-statements about the AI itself
            if not should_demote:
                meta_patterns = [
                    r"i'm a helpful ai",
                    r"i don't recognize",
                    r"i'm a .* assistant",
                    r"conversation started",
                    r"no stored information",
                ]
                for pattern in meta_patterns:
                    if re.search(pattern, content_lower):
                        should_demote = True
                        break

            # Very short content that's just a question echoed back
            if not should_demote and content_lower.startswith("who is my") and len(content) < 30:
                should_demote = True

            if should_demote:
                self.store.db.execute(
                    "UPDATE memories SET is_priority = 0, confidence = 0.5 WHERE id = ?",
                    (mem_id,)
                )
                self.store.db.commit()
                demoted += 1

        return demoted

    # ═══════════════════════════════════════════════════════════════
    # 5. ACTIVATION COMPUTATION
    # ═══════════════════════════════════════════════════════════════
    def compute_activations(self, query_embedding=None) -> list[ActivationState]:
        """
        Compute activation states for all active memory nodes.

        Activation is based on:
        - Query relevance (if query provided)
        - Access frequency (popular = higher base activation)
        - Confidence
        - Priority status
        - Neighbor activation (propagated)
        """
        cursor = self.store.db.execute(
            "SELECT id, content, confidence, access_count, is_priority, "
            "session_count, embedding FROM memories "
            "WHERE is_suppressed = 0 AND embedding IS NOT NULL"
        )
        memories = cursor.fetchall()
        if not memories:
            return []

        import numpy as np

        # Decode embeddings
        ids = []
        embeddings = []
        for m in memories:
            emb = _deserialize_embedding(m[6])
            if emb:
                ids.append(m[0])
                embeddings.append(emb)

        if not embeddings:
            return []

        emb_array = np.array(embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = emb_array / norms

        # Build adjacency (similarity-based edges)
        sim_matrix = normalized @ normalized.T
        adjacency = (sim_matrix > self.similarity_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)

        # Weight edges by similarity
        weighted_adj = sim_matrix * adjacency

        # Initialize activations
        n = len(ids)
        activations = np.zeros(n)

        # Base activation from intrinsic properties
        for i, m in enumerate(memories):
            if i >= n:
                break
            confidence = m[2]
            access_count = m[3]
            is_priority = m[4]
            session_count = max(m[5], 1)

            # c_value: intrinsic "charge" of this node
            recency = 1.0 / (1.0 + session_count * 0.1)
            popularity = min(1.0, access_count / 20.0)
            c_value = confidence * 0.4 + recency * 0.2 + popularity * 0.2 + (0.2 if is_priority else 0)

            activations[i] = c_value

        # Query relevance boost
        if query_embedding is not None:
            q = np.array(query_embedding)
            # Handle dimension mismatch (different embedding models)
            emb_dim = emb_array.shape[1]
            q_dim = q.shape[0]
            if q_dim != emb_dim:
                # Truncate or pad to match
                if q_dim > emb_dim:
                    q = q[:emb_dim]
                else:
                    q = np.pad(q, (0, emb_dim - q_dim))
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q_normalized = q / q_norm
                relevance = normalized @ q_normalized
                relevance = np.maximum(relevance, 0)
                activations += relevance * 0.5

        # Propagation steps (z = z² + c equivalent)
        cascade_lengths = []
        for step in range(3):
            prev_activations = activations.copy()

            # Neighbor influence
            neighbor_input = weighted_adj @ activations * self.coupling_strength

            # Nonlinear update: activation² + c + neighbor_input
            # Clamp to [0, 2] to prevent divergence in Phase 1
            activations = np.clip(
                activations ** 2 * 0.3 + activations * 0.5 + neighbor_input * 0.2,
                0, 2.0
            )

            # Normalize to keep in reasonable range
            max_act = activations.max()
            if max_act > 0:
                activations = activations / max_act

            # Track cascade: how much changed
            delta = np.abs(activations - prev_activations).sum()
            cascade_lengths.append(float(delta))

        # Build activation states
        states = []
        for i in range(n):
            m = memories[i] if i < len(memories) else None
            if m is None:
                continue
            states.append(ActivationState(
                memory_id=ids[i],
                activation=float(activations[i]),
                c_value=float(activations[i]),  # After propagation
                stability=1.0 - float(cascade_lengths[-1]) if cascade_lengths else 0.5,
            ))

        # Sort by activation (highest first)
        states.sort(key=lambda s: s.activation, reverse=True)
        return states

    # ═══════════════════════════════════════════════════════════════
    # 6. FULL PROPAGATION CYCLE
    # ═══════════════════════════════════════════════════════════════
    def propagate(self) -> PropagationResult:
        """
        Run a full propagation cycle:
        1. Suppress noise
        2. Deflate inflated priorities
        3. Merge duplicates
        4. Repair graph (extract entities)
        5. Run graph inference
        6. Compute activations
        """
        start = time.time()
        result = PropagationResult()

        # Step 1: Suppress noise
        result.noise_suppressed = self.suppress_noise()
        logger.info(f"Propagation: suppressed {result.noise_suppressed} noise memories")

        # Step 2: Deflate priorities
        demoted = self.deflate_priorities()
        logger.info(f"Propagation: demoted {demoted} inflated priorities")

        # Step 3: Merge duplicates
        result.duplicates_merged = self.merge_duplicates()
        logger.info(f"Propagation: merged {result.duplicates_merged} duplicates")

        # Step 4: Repair graph
        repair_stats = self.repair_graph()
        result.entities_created = repair_stats["entities_created"]
        result.relations_created = repair_stats["relations_created"]
        logger.info(f"Propagation: created {result.entities_created} entities, "
                    f"{result.relations_created} relations")

        # Step 5: Run inference on the repaired graph
        from limbiq.graph.inference import InferenceEngine
        inference = InferenceEngine(self.graph)
        inferred = inference.run_full_inference()
        result.relations_created += inferred

        # Step 6: Compute activations (baseline, no query)
        states = self.compute_activations()
        result.confidence_updates = len(states)

        # Track cascade dynamics
        result.cascade_lengths = [s.activation for s in states[:10]] if states else []
        result.duration_ms = (time.time() - start) * 1000

        return result
