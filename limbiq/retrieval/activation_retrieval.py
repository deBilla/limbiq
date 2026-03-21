"""
Activation-Weighted Retrieval — Phase 4
========================================
Replaces pure embedding similarity with a hybrid scoring function:

  final_score = α * embedding_similarity + β * gnn_activation + γ * graph_relevance

Where:
- embedding_similarity: cosine similarity between query and memory (existing)
- gnn_activation: Phase 2 GNN's learned activation for this memory
- graph_relevance: boost for memories connected to query-relevant entities

This is the bridge between the graph intelligence (Phases 1-3) and the
LLM's context window. Without this, the graph improvements don't flow
into actual LLM behavior.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """A memory with breakdown of how it scored."""
    memory_id: str
    content: str
    final_score: float
    embedding_sim: float
    activation: float
    graph_boost: float
    is_priority: bool
    tier: str


class ActivationRetrieval:
    """
    Hybrid retrieval combining embedding similarity with GNN activations
    and knowledge graph structure.
    """

    def __init__(self, store, graph, embedding_engine,
                 gnn_propagation=None, user_name: str = "Dimuthu"):
        self.store = store
        self.graph = graph
        self.embeddings = embedding_engine
        self.gnn = gnn_propagation
        self.user_name = user_name

        # Scoring weights
        self.alpha = 0.45   # embedding similarity weight
        self.beta = 0.35    # GNN activation weight
        self.gamma = 0.20   # graph entity relevance weight

    def search(self, query: str, query_embedding=None,
               top_k: int = 10, include_suppressed: bool = False) -> list[ScoredMemory]:
        """
        Retrieve memories using activation-weighted scoring.

        Steps:
        1. Compute embedding similarity (existing path)
        2. Get GNN activations biased toward query
        3. Compute graph entity relevance boost
        4. Blend scores and return top-k
        """
        if query_embedding is None:
            query_embedding = self.embeddings.embed(query)

        # Step 1: Get all candidate memories with embedding similarity
        from limbiq.store.memory_store import _deserialize_embedding
        clause = "" if include_suppressed else "WHERE is_suppressed = 0"
        cursor = self.store.db.execute(
            f"SELECT id, content, tier, confidence, is_priority, is_suppressed, "
            f"embedding FROM memories {clause}"
        )

        candidates = []
        for row in cursor.fetchall():
            emb = _deserialize_embedding(row[6])
            if emb is None:
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in emb) ** 0.5
            sim = dot / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0

            candidates.append({
                "id": row[0],
                "content": row[1],
                "tier": row[2],
                "confidence": row[3],
                "is_priority": bool(row[4]),
                "embedding_sim": max(0, sim),
            })

        if not candidates:
            return []

        # Step 2: Get GNN activations (query-biased)
        activation_map = {}
        if self.gnn:
            try:
                activations = self.gnn.compute_activations(query_embedding)
                activation_map = {mid: act for mid, act in activations}
            except Exception as e:
                logger.warning(f"GNN activation failed: {e}")

        # Step 3: Compute graph entity relevance
        graph_boost_map = self._compute_graph_boosts(query)

        # Step 4: Blend scores
        scored = []
        for c in candidates:
            emb_score = c["embedding_sim"]
            act_score = activation_map.get(c["id"], 0.1)  # Default low activation
            graph_score = graph_boost_map.get(c["id"], 0.0)

            # Priority memories get a floor activation
            if c["is_priority"] and act_score < 0.3:
                act_score = 0.3

            # Confidence scaling — high confidence memories get slight boost
            conf_mult = 0.8 + 0.2 * c["confidence"]

            final = (
                self.alpha * emb_score +
                self.beta * act_score +
                self.gamma * graph_score
            ) * conf_mult

            scored.append(ScoredMemory(
                memory_id=c["id"],
                content=c["content"],
                final_score=final,
                embedding_sim=emb_score,
                activation=act_score,
                graph_boost=graph_score,
                is_priority=c["is_priority"],
                tier=c["tier"],
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored[:top_k]

    def _compute_graph_boosts(self, query: str) -> dict:
        """
        Boost memories connected to entities mentioned in the query.

        If query mentions "Prabhashi", memories containing facts about
        Prabhashi or connected entities get a boost.
        """
        boosts = {}

        # Find entities mentioned in the query
        query_lower = query.lower()
        entities = self.graph.get_all_entities()
        mentioned_entity_ids = set()

        for e in entities:
            if e.name.lower() in query_lower:
                mentioned_entity_ids.add(e.id)

        if not mentioned_entity_ids:
            return boosts

        # Get all relations involving mentioned entities
        related_entity_ids = set()
        for eid in mentioned_entity_ids:
            relations = self.graph.get_relations_for(eid)
            for r in relations:
                related_entity_ids.add(r.subject_id)
                related_entity_ids.add(r.object_id)

        # Get entity names for content matching
        entity_names = set()
        for e in entities:
            if e.id in mentioned_entity_ids:
                entity_names.add(e.name.lower())
            if e.id in related_entity_ids:
                entity_names.add(e.name.lower())

        # Scan memories for mentions of these entities
        clause = "WHERE is_suppressed = 0"
        rows = self.store.db.execute(
            f"SELECT id, content FROM memories {clause}"
        ).fetchall()

        for mid, content in rows:
            content_lower = content.lower()
            # Direct mention of queried entity
            direct = sum(1 for name in entity_names
                        if name in content_lower and len(name) > 2)
            if direct > 0:
                # Scale: 1 mention = 0.5, 2+ = 0.8, capped at 1.0
                boosts[mid] = min(1.0, 0.3 + direct * 0.25)

        return boosts


# ─── Graph-Aware Context Builder ─────────────────────────────────

class GraphStateContextBuilder:
    """
    Build structured context for the LLM from graph state instead of
    raw memory text. This is what the LLM actually sees.

    Phase 4 improvement: instead of dumping flat memory strings, we
    produce structured graph-state context that's more token-efficient
    and easier for small models to parse.
    """

    def __init__(self, graph, inference_engine):
        self.graph = graph
        self.inference = inference_engine

    def build_context(self, query: str, scored_memories: list[ScoredMemory],
                      world_summary: str = None, graph_answer: str = None,
                      active_rules=None, caution_flag: str = None) -> str:
        """Build the full context string for LLM injection."""
        sections = []

        # Caution flag
        if caution_flag:
            sections.append(
                f"[CAUTION: {caution_flag}]\n"
                "Double-check claims against memory. Say so if unsure."
            )

        # Graph direct answer (most token-efficient)
        if graph_answer:
            sections.append(f"[KNOWN FACT] {graph_answer}")

        # World summary from graph
        if world_summary:
            sections.append(f"[ABOUT YOU] {world_summary}")

        # Entity-focused context: if the query mentions specific entities,
        # include their graph descriptions
        entity_context = self._entity_context_for_query(query)
        if entity_context and entity_context not in (world_summary or ""):
            sections.append(f"[ENTITY DETAILS] {entity_context}")

        # Behavioral rules
        if active_rules:
            rules_text = "; ".join(
                rule.rule_text for rule in active_rules
            )
            sections.append(f"[STYLE] {rules_text}")

        # Top scored memories (only those not already covered by graph)
        if scored_memories:
            world_lower = ((world_summary or "") + " " + (graph_answer or "")).lower()
            ungraphed = []
            for sm in scored_memories:
                if not self._is_covered(sm.content, world_lower):
                    ungraphed.append(sm)

            # Priority memories first
            priority = [m for m in ungraphed if m.is_priority][:3]
            relevant = [m for m in ungraphed if not m.is_priority][:3]

            if priority:
                facts = "\n".join(
                    f"  - {m.content}" for m in priority
                )
                sections.append(f"[IMPORTANT]\n{facts}")

            # When graph provides context, cap relevant memories
            max_relevant = 2 if (world_summary or graph_answer) else 4
            if relevant:
                items = "\n".join(
                    f"  - [{m.final_score:.0%}] {m.content}"
                    for m in relevant[:max_relevant]
                )
                sections.append(f"[RELEVANT]\n{items}")

        if not sections:
            return ""

        context = "\n\n".join(sections)
        return f"<memory_context>\n{context}\n</memory_context>"

    def _entity_context_for_query(self, query: str) -> str:
        """Get entity descriptions relevant to the query."""
        query_lower = query.lower()
        entities = self.graph.get_all_entities()
        descriptions = []

        for e in entities:
            if e.name.lower() in query_lower and len(e.name) > 2:
                desc = self.inference.describe_entity(e.name)
                if desc:
                    descriptions.append(desc)

        return " ".join(descriptions) if descriptions else ""

    @staticmethod
    def _is_covered(content: str, summary_lower: str) -> bool:
        """Check if memory content is already in the graph summary."""
        if not summary_lower:
            return False
        words = content.split()
        proper_nouns = [w.strip(".,!?'\"") for w in words
                       if w[0:1].isupper() and len(w) > 2]
        if not proper_nouns:
            return False
        found = sum(1 for pn in proper_nouns if pn.lower() in summary_lower)
        return found >= len(proper_nouns) * 0.5


# ─── Enhanced LoRA Training Data ─────────────────────────────────

class GraphTrainingDataGenerator:
    """
    Generate LoRA training data that teaches the LLM to read
    graph-structured context instead of raw memory dumps.

    Produces instruction-tuning pairs:
      Input:  [graph context] + user question
      Output: answer that correctly uses graph information

    This trains the 8B model to understand and leverage the
    structured memory format.
    """

    def __init__(self, graph, inference_engine, store_db,
                 user_name: str = "Dimuthu"):
        self.graph = graph
        self.inference = inference_engine
        self.store_db = store_db
        self.user_name = user_name

    def generate(self) -> list[dict]:
        """
        Generate training pairs from the knowledge graph.
        Returns list of {"prompt": ..., "completion": ...} dicts.
        """
        pairs = []

        entities = self.graph.get_all_entities()
        entities_by_id = {e.id: e for e in entities}
        all_relations = self.graph.get_all_relations(include_inferred=True)

        world_summary = self.inference.get_user_world(self.user_name)

        # Type 1: Direct entity questions
        for rel in all_relations:
            if rel.is_inferred:
                continue
            subj = entities_by_id.get(rel.subject_id)
            obj = entities_by_id.get(rel.object_id)
            if not subj or not obj:
                continue

            if subj.name == self.user_name:
                # Generate Q&A about the user's relations
                q, a = self._make_relation_qa(rel.predicate, obj.name)
                if q and a:
                    pairs.append({
                        "prompt": self._wrap_with_context(q, world_summary),
                        "completion": a,
                    })

        # Type 2: Inference questions (tests if model can use inferred relations)
        for rel in all_relations:
            if not rel.is_inferred:
                continue
            subj = entities_by_id.get(rel.subject_id)
            obj = entities_by_id.get(rel.object_id)
            if not subj or not obj:
                continue

            q, a = self._make_inference_qa(subj.name, rel.predicate, obj.name)
            if q and a:
                pairs.append({
                    "prompt": self._wrap_with_context(q, world_summary),
                    "completion": a,
                })

        # Type 3: "What do you know about me?" — tests world summary comprehension
        if world_summary:
            pairs.append({
                "prompt": self._wrap_with_context(
                    "What do you know about me?", world_summary
                ),
                "completion": f"Based on what I remember: {world_summary}",
            })

        # Type 4: Negation / boundary questions — tests that model doesn't hallucinate
        pairs.extend(self._make_negation_pairs(world_summary))

        return pairs

    def _make_relation_qa(self, predicate: str, object_name: str):
        """Generate a question-answer pair for a direct relation."""
        templates = {
            "wife": (f"Who is my wife?", f"Your wife is {object_name}."),
            "husband": (f"Who is my husband?", f"Your husband is {object_name}."),
            "father": (f"Who is my father?", f"Your father is {object_name}."),
            "mother": (f"Who is my mother?", f"Your mother is {object_name}."),
            "sister": (f"Do I have a sister named {object_name}?",
                       f"Yes, {object_name} is your sister."),
            "brother": (f"Do I have a brother named {object_name}?",
                        f"Yes, {object_name} is your brother."),
            "works_at": (f"Where do I work?", f"You work at {object_name}."),
            "role": (f"What do I do for work?", f"You are a {object_name}."),
        }
        return templates.get(predicate, (None, None))

    def _make_inference_qa(self, subject_name: str, predicate: str, object_name: str):
        """Generate Q&A for inferred relations."""
        templates = {
            "father_in_law_of": (
                f"What is {subject_name}'s relationship to {object_name}?",
                f"{subject_name} is {object_name}'s father-in-law."
            ),
            "mother_in_law_of": (
                f"What is {subject_name}'s relationship to {object_name}?",
                f"{subject_name} is {object_name}'s mother-in-law."
            ),
            "sister_in_law_of": (
                f"What is {subject_name}'s relationship to {object_name}?",
                f"{subject_name} is {object_name}'s sister-in-law."
            ),
            "brother_in_law_of": (
                f"What is {subject_name}'s relationship to {object_name}?",
                f"{subject_name} is {object_name}'s brother-in-law."
            ),
        }
        return templates.get(predicate, (None, None))

    def _make_negation_pairs(self, world_summary: str) -> list[dict]:
        """Generate pairs that teach the model to say 'I don't know'."""
        pairs = []

        negation_questions = [
            ("Do I have a brother?", "I don't have any record of you having a brother."),
            ("What's my dog's name?", "I don't have any information about a dog."),
            ("Where did I go to school?", "I don't have that information in my memory."),
        ]

        for q, a in negation_questions:
            pairs.append({
                "prompt": self._wrap_with_context(q, world_summary),
                "completion": a,
            })

        return pairs

    def _wrap_with_context(self, question: str, world_summary: str) -> str:
        """Wrap a question with graph-state context (training format)."""
        return (
            f"Here is what you remember about me from past conversations "
            f"(use this naturally, do NOT mention section names or tags). "
            f"Answer using ONLY this memory — do NOT search the web:\n\n"
            f"<memory_context>\n"
            f"[ABOUT YOU] {world_summary}\n"
            f"</memory_context>\n\n"
            f"Now answer this: {question}"
        )

    def export_jsonl(self, output_path: str) -> int:
        """Export training data as JSONL for LoRA training."""
        pairs = self.generate()

        import json
        with open(output_path, 'w') as f:
            for pair in pairs:
                entry = {
                    "messages": [
                        {"role": "user", "content": pair["prompt"]},
                        {"role": "assistant", "content": pair["completion"]},
                    ]
                }
                f.write(json.dumps(entry) + "\n")

        return len(pairs)
