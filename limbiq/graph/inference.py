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

# Co-parent pairs: if user has both of these relations, the targets are spouses
# e.g., Dimuthu→father→Upananda + Dimuthu→mother→Renuka → Upananda married_to Renuka
CO_PARENT_PAIRS = [
    ("father", "mother"),  # user's parents are married
]

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

        all_relations = self.graph.get_all_relations(include_inferred=False)
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
