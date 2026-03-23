"""
Deterministic relationship inference.

Given explicit relations in the graph, compute implied relations.
Pure logic — no LLM needed. Fast and predictable.

Example:
  Dimuthu → father → Upananda
  Dimuthu → wife → Prabhashi
  ∴ Upananda → father_in_law_of → Prabhashi  (inferred)
"""

from limbiq.graph.store import GraphStore, Relation, Entity


# (A_to_B_relation, A_to_C_relation) → B_to_C_relation
# Both relations share the same subject (the user).
#
# IMPORTANT: Only in-law rules work with same-subject pairs.
# Grandparent/uncle/aunt rules CANNOT work here because they require
# multi-hop traversal across entities, not two outgoing edges from one node.
#
# Example that works:
#   Dimuthu→father→Upananda + Dimuthu→wife→Prabhashi
#   → Upananda is father_in_law_of Prabhashi  ✓
#
# Example that BREAKS (removed):
#   Dimuthu→father→Upananda + Dimuthu→mother→Renuka
#   → ("father","mother") was mapped to "grandfather_of"
#   → Upananda grandfather_of Renuka — WRONG, they're co-parents!
#
# Co-parent detection: when two family relations share a subject,
# the objects may be spouses (e.g., father + mother = married couple).
INFERENCE_RULES = {
    # In-law relationships (via marriage) — these work because
    # the two targets are from different families
    ("father", "wife"): "father_in_law_of",
    ("mother", "wife"): "mother_in_law_of",
    ("father", "husband"): "father_in_law_of",
    ("mother", "husband"): "mother_in_law_of",
    ("brother", "wife"): "brother_in_law_of",
    ("sister", "husband"): "sister_in_law_of",
    ("brother", "husband"): "brother_in_law_of",
    ("sister", "wife"): "sister_in_law_of",
}

# Reverse in-law rules: given user→wife→X and user→father_in_law→Y,
# infer X→father→Y (wife's father = father-in-law).
# Format: (user_rel_to_spouse, user_rel_to_inlaw) → (spouse, inlaw_pred, inlaw)
REVERSE_INLAW_RULES = {
    ("wife", "father_in_law"): "father",       # wife's father = father-in-law
    ("wife", "mother_in_law"): "mother",        # wife's mother = mother-in-law
    ("husband", "father_in_law"): "father",     # husband's father = father-in-law
    ("husband", "mother_in_law"): "mother",     # husband's mother = mother-in-law
    ("wife", "brother_in_law"): "brother",      # wife's brother = brother-in-law
    ("wife", "sister_in_law"): "sister",        # wife's sister = sister-in-law
    ("husband", "brother_in_law"): "brother",
    ("husband", "sister_in_law"): "sister",
}

# Co-parent pairs: if user has both of these relations, the targets are spouses
# e.g., Dimuthu→father→Upananda + Dimuthu→mother→Renuka → Upananda married_to Renuka
CO_PARENT_PAIRS = [
    ("father", "mother"),  # user's parents are married
]

# Multi-hop rules: traverse across entities
# A→pred1→B, B→pred2→C ⟹ C is inferred_pred of A
MULTI_HOP_RULES = {
    # Grandparent relationships (both parents + both genders)
    ("father", "father"): "grandfather_of",
    ("father", "mother"): "grandmother_of",
    ("mother", "father"): "grandfather_of",
    ("mother", "mother"): "grandmother_of",
    # Uncle/aunt relationships (parent + sibling)
    ("father", "brother"): "uncle_of",
    ("father", "sister"): "aunt_of",
    ("mother", "brother"): "uncle_of",
    ("mother", "sister"): "aunt_of",
}

# Human-readable descriptions for context injection
RELATION_DESCRIPTIONS = {
    "father": "{object} is {subject}'s father",
    "mother": "{object} is {subject}'s mother",
    "wife": "{object} is {subject}'s wife",
    "husband": "{object} is {subject}'s husband",
    "brother": "{object} is {subject}'s brother",
    "sister": "{object} is {subject}'s sister",
    "son": "{object} is {subject}'s son",
    "daughter": "{object} is {subject}'s daughter",
    "works_at": "{subject} works at {object}",
    "lives_in": "{subject} lives in {object}",
    "role": "{subject} is a {object}",
    "father_in_law_of": "{subject} is {object}'s father-in-law",
    "mother_in_law_of": "{subject} is {object}'s mother-in-law",
    "brother_in_law_of": "{subject} is {object}'s brother-in-law",
    "sister_in_law_of": "{subject} is {object}'s sister-in-law",
    "grandfather_of": "{subject} is {object}'s grandfather",
    "grandmother_of": "{subject} is {object}'s grandmother",
    "uncle_of": "{subject} is {object}'s uncle",
    "aunt_of": "{subject} is {object}'s aunt",
    "married_to": "{subject} and {object} are married",
    "colleague": "{subject} and {object} are colleagues",
    "friend": "{subject} and {object} are friends",
    "partner": "{object} is {subject}'s partner",
    "boss": "{object} is {subject}'s boss",
    "manager": "{object} is {subject}'s manager",
    "interested_in": "{subject} is interested in {object}",
}


class InferenceEngine:
    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run_full_inference(self) -> int:
        """
        Compute all inferable relations from the current graph.
        Clears previous inferences first (they'll be recomputed).
        Returns the number of new relations inferred.
        """
        self.graph.remove_inferred()

        # Clean junk entities BEFORE inference to prevent combinatorial explosion
        self.graph._cleanup_junk_entities()

        all_relations = self.graph.get_all_relations(include_inferred=False)

        # Build set of valid entity IDs (skip junk entities)
        valid_entity_ids = set()
        for e in self.graph.get_all_entities():
            if not self.graph._is_junk_name(e.name):
                valid_entity_ids.add(e.id)

        # Filter relations to only those between valid entities AND
        # high confidence (>= 0.8) explicit relations. Low-confidence relations
        # from noisy extraction should not seed inference chains.
        all_relations = [
            r for r in all_relations
            if r.subject_id in valid_entity_ids
            and r.object_id in valid_entity_ids
            and r.confidence >= 0.8
        ]

        inferred_count = 0

        # 1. In-law inference: for pairs of relations sharing the same subject,
        #    check if we can infer an in-law relationship between their objects.
        for rel_a in all_relations:
            for rel_b in all_relations:
                if rel_a.id == rel_b.id:
                    continue
                if rel_a.subject_id != rel_b.subject_id:
                    continue

                key = (rel_a.predicate, rel_b.predicate)
                inferred_predicate = INFERENCE_RULES.get(key)
                if not inferred_predicate:
                    continue

                # Don't create self-relations
                if rel_a.object_id == rel_b.object_id:
                    continue

                # Check if this relation already exists (explicit or inferred)
                existing = self.graph.db.execute(
                    "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                    (rel_a.object_id, inferred_predicate, rel_b.object_id)
                ).fetchone()
                if existing:
                    continue

                new_rel = Relation(
                    subject_id=rel_a.object_id,
                    predicate=inferred_predicate,
                    object_id=rel_b.object_id,
                    confidence=min(rel_a.confidence, rel_b.confidence) * 0.9,
                    is_inferred=True,
                )
                self.graph.add_relation(new_rel)
                inferred_count += 1

        # 2. Co-parent inference: if user has (father→X, mother→Y),
        #    X and Y are likely married.
        for father_pred, mother_pred in CO_PARENT_PAIRS:
            # Group by subject
            by_subject = {}
            for r in all_relations:
                by_subject.setdefault(r.subject_id, []).append(r)

            for subject_id, rels in by_subject.items():
                fathers = [r for r in rels if r.predicate == father_pred]
                mothers = [r for r in rels if r.predicate == mother_pred]
                for f in fathers:
                    for m in mothers:
                        if f.object_id == m.object_id:
                            continue
                        # Check if spouse relation already exists in either direction
                        existing = self.graph.db.execute(
                            "SELECT id FROM relations WHERE "
                            "(subject_id=? AND object_id=?) OR (subject_id=? AND object_id=?)",
                            (f.object_id, m.object_id, m.object_id, f.object_id)
                        ).fetchone()
                        if not existing:
                            self.graph.add_relation(Relation(
                                subject_id=f.object_id,
                                predicate="married_to",
                                object_id=m.object_id,
                                confidence=min(f.confidence, m.confidence) * 0.85,
                                is_inferred=True,
                            ))
                            inferred_count += 1

        # 3. Reverse in-law inference: given user→wife→Prabhashi and
        #    user→father_in_law→Chandrasiri, infer Prabhashi→father→Chandrasiri.
        #    This connects the graph through intermediate family relationships.
        by_subject = {}
        for r in all_relations:
            by_subject.setdefault(r.subject_id, []).append(r)

        for subject_id, rels in by_subject.items():
            for rel_spouse in rels:
                for rel_inlaw in rels:
                    if rel_spouse.id == rel_inlaw.id:
                        continue
                    key = (rel_spouse.predicate, rel_inlaw.predicate)
                    direct_pred = REVERSE_INLAW_RULES.get(key)
                    if not direct_pred:
                        continue
                    # spouse→direct_pred→inlaw (e.g., Prabhashi→father→Chandrasiri)
                    if rel_spouse.object_id == rel_inlaw.object_id:
                        continue
                    existing = self.graph.db.execute(
                        "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                        (rel_spouse.object_id, direct_pred, rel_inlaw.object_id)
                    ).fetchone()
                    if not existing:
                        self.graph.add_relation(Relation(
                            subject_id=rel_spouse.object_id,
                            predicate=direct_pred,
                            object_id=rel_inlaw.object_id,
                            confidence=min(rel_spouse.confidence, rel_inlaw.confidence) * 0.85,
                            is_inferred=True,
                        ))
                        inferred_count += 1

        # 4. Multi-hop inference: traverse across entities
        #    A→pred1→B, B→pred2→C ⟹ infer C is inferred_pred of A
        inferred_count += self._multi_hop_inference(all_relations)

        return inferred_count

    def _multi_hop_inference(self, all_relations: list) -> int:
        """
        Multi-hop inference: traverse across entities.
        A→pred1→B, B→pred2→C ⟹ infer C is inferred_pred of A

        GUARD: Only uses high-confidence explicit relations for hops.
        Low-confidence relations (< 0.8) are often extraction noise
        and should not be used to build inference chains.

        Returns the number of new relations inferred.
        """
        # Only use high-confidence explicit (non-inferred) relations for hops
        # This prevents chains built on noisy extraction results
        hop_relations = [
            r for r in all_relations
            if r.confidence >= 0.8 and not r.is_inferred
        ]

        # Build adjacency from high-confidence relations only
        outgoing = {}
        for r in hop_relations:
            outgoing.setdefault(r.subject_id, []).append(
                (r.predicate, r.object_id, r.confidence)
            )

        inferred_count = 0
        max_inferred = 20  # Safety cap

        for entity_a, first_hops in outgoing.items():
            for pred1, entity_b, conf1 in first_hops:
                if entity_b not in outgoing:
                    continue
                for pred2, entity_c, conf2 in outgoing[entity_b]:
                    if entity_c == entity_a:
                        continue

                    key = (pred1, pred2)
                    inferred_pred = MULTI_HOP_RULES.get(key)
                    if not inferred_pred:
                        continue

                    existing = self.graph.db.execute(
                        "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                        (entity_c, inferred_pred, entity_a)
                    ).fetchone()
                    if existing:
                        continue

                    new_rel = Relation(
                        subject_id=entity_c,
                        predicate=inferred_pred,
                        object_id=entity_a,
                        confidence=min(conf1, conf2) * 0.8,
                        is_inferred=True,
                    )
                    self.graph.add_relation(new_rel)
                    inferred_count += 1

                    if inferred_count >= max_inferred:
                        return inferred_count

        return inferred_count

    def query_relationship(self, entity_a_name: str, entity_b_name: str) -> dict:
        """
        Find the relationship between two entities.
        Returns both direct and inferred relationships.
        """
        entity_a = self.graph.find_entity_by_name(entity_a_name)
        entity_b = self.graph.find_entity_by_name(entity_b_name)

        if not entity_a or not entity_b:
            return {"found": False, "relations": [],
                    "entity_a": entity_a, "entity_b": entity_b}

        rows = self.graph.db.execute(
            "SELECT * FROM relations WHERE "
            "(subject_id=? AND object_id=?) OR (subject_id=? AND object_id=?)",
            (entity_a.id, entity_b.id, entity_b.id, entity_a.id),
        ).fetchall()

        all_entities = {e.id: e for e in self.graph.get_all_entities()}
        results = []
        for row in rows:
            rel = self.graph._row_to_relation(row)
            subj_name = all_entities.get(rel.subject_id, Entity(name="?")).name
            obj_name = all_entities.get(rel.object_id, Entity(name="?")).name

            template = RELATION_DESCRIPTIONS.get(
                rel.predicate, "{subject} is related to {object}")
            description = template.format(subject=subj_name, object=obj_name)

            results.append({
                "predicate": rel.predicate,
                "is_inferred": rel.is_inferred,
                "description": description,
                "confidence": rel.confidence,
            })

        return {"found": len(results) > 0, "relations": results,
                "entity_a": entity_a, "entity_b": entity_b}

    def describe_entity(self, entity_name: str) -> str:
        """
        Natural language summary of everything known about an entity.
        This is what gets injected into context — NOT raw graph data.
        """
        entity = self.graph.find_entity_by_name(entity_name)
        if not entity:
            return ""

        relations = self.graph.get_relations_for(entity.id)
        if not relations:
            return ""

        all_entities = {e.id: e for e in self.graph.get_all_entities()}
        descriptions = []

        for rel in relations:
            subj_name = all_entities.get(rel.subject_id, Entity(name="?")).name
            obj_name = all_entities.get(rel.object_id, Entity(name="?")).name

            template = RELATION_DESCRIPTIONS.get(rel.predicate)
            if template:
                desc = template.format(subject=subj_name, object=obj_name)
                if rel.is_inferred:
                    desc += " (inferred)"
                descriptions.append(desc)

        return "; ".join(descriptions) + "." if descriptions else ""

    def get_user_world(self, user_name: str) -> str:
        """
        Compact summary of the user's known world.
        Replaces raw memory dumps in context.

        Instead of 5 separate memory strings (~200 tokens):
            "User's father is Upananda"
            "User's wife is Prabhashi"

        Returns one connected summary (~40 tokens):
            "Your father is Upananda (Prabhashi's father-in-law).
             Your wife is Prabhashi."
        """
        user = self.graph.find_entity_by_name(user_name)
        if not user:
            return ""

        relations = self.graph.get_relations_for(user.id)
        if not relations:
            return ""

        all_entities = {e.id: e for e in self.graph.get_all_entities()}

        family_preds = {"father", "mother", "wife", "husband", "brother", "sister",
                        "son", "daughter", "partner"}
        work_preds = {"works_at", "role", "colleague", "manages", "boss", "manager"}
        location_preds = {"lives_in", "based_in"}

        family = []
        work = []
        location = []
        other = []

        for rel in relations:
            if rel.subject_id != user.id:
                continue  # Only relations FROM the user

            obj_name = all_entities.get(rel.object_id, Entity(name="?")).name

            if rel.predicate in family_preds:
                # Check for inferred relations for this family member
                family_member_rels = [
                    r for r in self.graph.get_relations_for(rel.object_id)
                    if r.is_inferred and r.subject_id == rel.object_id
                ]
                extras = []
                for fr in family_member_rels:
                    other_name = all_entities.get(fr.object_id, Entity(name="?")).name
                    template = RELATION_DESCRIPTIONS.get(fr.predicate, "")
                    if template:
                        extras.append(template.format(subject=obj_name, object=other_name))

                if extras:
                    family.append(f"Your {rel.predicate} is {obj_name} ({'; '.join(extras)})")
                else:
                    family.append(f"Your {rel.predicate} is {obj_name}")

            elif rel.predicate in work_preds:
                if rel.predicate == "role":
                    work.append(f"You are a {obj_name}")
                else:
                    work.append(f"You {rel.predicate.replace('_', ' ')} {obj_name}")
            elif rel.predicate in location_preds:
                location.append(f"You live in {obj_name}")
            else:
                other.append(f"Your {rel.predicate}: {obj_name}")

        parts = []
        if family:
            parts.append(". ".join(family))
        if work:
            parts.append(". ".join(work))
        if location:
            parts.append(". ".join(location))
        if other:
            parts.append(". ".join(other))

        return ". ".join(parts) + "." if parts else ""

    def get_relevant_graph_context(
        self,
        query: str,
        embedding_engine=None,
        top_k: int = 5,
    ) -> str:
        """
        Given a user query, find the most relevant entities using embedding
        similarity and return their relationships as structured context.

        This is the MAIN injection point for graph knowledge into LLM context.
        Unlike graph_query.try_answer() (regex-based, narrow), this works for
        ANY query by finding semantically related entities.

        Returns a compact string of graph triples, or "" if no relevant data.
        """
        entities = self.graph.get_all_entities()
        if not entities or not embedding_engine:
            return ""

        # Score entities by embedding similarity to the query
        query_emb = embedding_engine.embed(query)
        scored = []
        for e in entities:
            if self.graph._is_junk_name(e.name):
                continue
            e_emb = embedding_engine.embed(e.name)
            sim = embedding_engine.similarity(query_emb, e_emb)
            scored.append((e, sim))

        # Also check if entity name appears directly in query (strong signal)
        query_lower = query.lower()
        for i, (e, sim) in enumerate(scored):
            if e.name.lower() in query_lower:
                scored[i] = (e, max(sim, 0.95))  # Boost to top

        scored.sort(key=lambda x: x[1], reverse=True)
        top_entities = [e for e, sim in scored[:top_k] if sim > 0.3]

        if not top_entities:
            return ""

        # Gather all relations involving these entities
        entity_map = {e.id: e.name for e in entities}
        top_ids = {e.id for e in top_entities}

        all_rels = self.graph.get_all_relations(include_inferred=True)
        relevant_rels = [
            r for r in all_rels
            if r.subject_id in top_ids or r.object_id in top_ids
        ]

        if not relevant_rels:
            return ""

        # Deduplicate: keep highest-confidence version of each (subj, pred, obj)
        seen = {}
        for r in relevant_rels:
            key = (r.subject_id, r.predicate, r.object_id)
            if key not in seen or r.confidence > seen[key].confidence:
                seen[key] = r

        # Format as readable triples, sorted by confidence
        triples = []
        for r in sorted(seen.values(), key=lambda x: -x.confidence):
            subj = entity_map.get(r.subject_id, "?")
            obj = entity_map.get(r.object_id, "?")
            template = RELATION_DESCRIPTIONS.get(r.predicate)
            if template:
                desc = template.format(subject=subj, object=obj)
            else:
                desc = f"{subj} {r.predicate.replace('_', ' ')} {obj}"
            if r.is_inferred:
                desc += f" (inferred, {r.confidence:.0%})"
            triples.append(desc)

        if not triples:
            return ""

        return "; ".join(triples[:10]) + "."
