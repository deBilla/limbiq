"""
Extract entities and relationships from text memories.

Three-tier extraction pipeline:
1. spaCy NER (PRIMARY) — finds entity names + types from natural language
2. Regex patterns (SECONDARY) — detects relationship predicates between entities
3. LLM (TERTIARY) — deferred batch extraction for complex cases

spaCy finds the WHAT (entity names), regex finds the HOW (relationships).
"""

import re
import logging
from typing import Optional

from limbiq.graph.store import Entity, Relation, GraphStore

logger = logging.getLogger(__name__)


# ── spaCy setup (lazy-loaded) ────────────────────────────────

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model. Falls back gracefully if not installed."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded for entity extraction")
        return _nlp
    except (ImportError, OSError) as e:
        logger.info(f"spaCy not available ({e}), using regex-only extraction")
        _nlp = False  # Mark as unavailable
        return None


# ── spaCy label → limbiq entity_type mapping ─────────────────

SPACY_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "company",
    "GPE": "place",       # Geo-political entity (country, city)
    "LOC": "place",       # Non-GPE location
    "FAC": "place",       # Facility
    "PRODUCT": "concept",
    "EVENT": "concept",
    "WORK_OF_ART": "concept",
    "NORP": "concept",    # Nationalities, religious groups
    "LANGUAGE": "concept",
}

# Labels to skip entirely
SPACY_SKIP_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


# ── Valid predicates ─────────────────────────────────────────

VALID_PREDICATES = {
    "father", "mother", "wife", "husband", "partner", "brother", "sister",
    "son", "daughter", "grandfather", "grandmother",
    "father_in_law", "mother_in_law", "brother_in_law", "sister_in_law",
    "son_in_law", "daughter_in_law",
    "step_father", "step_mother", "uncle", "aunt", "cousin",
    "works_at", "role", "colleague", "boss", "manager", "employee",
    "friend", "neighbor", "classmate",
    "lives_in", "based_in", "from",
    "pet", "dog", "cat",
    "married_to", "parent", "sibling", "child",
    "interested_in", "studies", "hobby",
}

RELATION_ALIASES = {
    "dad": "father", "mom": "mother", "mum": "mother",
    "pa": "father", "ma": "mother",
    "bro": "brother", "sis": "sister",
    "father in law": "father_in_law", "father-in-law": "father_in_law",
    "mother in law": "mother_in_law", "mother-in-law": "mother_in_law",
    "brother in law": "brother_in_law", "brother-in-law": "brother_in_law",
    "sister in law": "sister_in_law", "sister-in-law": "sister_in_law",
    "step father": "step_father", "step-father": "step_father",
    "step mother": "step_mother", "step-mother": "step_mother",
    "stepfather": "step_father", "stepmother": "step_mother",
}

INVALID_ENTITY_NAMES = {
    "none", "null", "n/a", "unknown", "undefined", "default", "",
    "?", "user", "the user", "assistant", "ai", "bot",
    "father", "mother", "wife", "husband", "partner", "brother", "sister",
    "son", "daughter", "dog", "cat", "pet", "boss", "friend", "colleague",
    "father-in-law", "mother-in-law", "step-father", "step-mother",
    "has", "have", "had", "was", "were", "is", "are", "been",
    "doing", "feeling", "going", "getting", "being", "having",
    "named", "called", "told", "said", "mentioned",
    "new", "old", "big", "small", "good", "bad", "great",
    "work", "home", "school", "office", "hospital", "doctor",
    "condition", "schedule", "care", "feeding", "well-being",
    "advice", "help", "question", "answer", "problem", "issue",
    "who", "what", "where", "when", "how", "which", "whom", "whose", "why",
    "talking", "working", "living", "asking", "telling", "saying", "looking",
    "the", "and", "but", "not", "also", "just", "very", "really",
    "about", "with", "from", "that", "this", "will", "would", "could",
    "other", "another", "some", "any", "all",
}


def _normalize_predicate(pred: str) -> str:
    p = pred.lower().strip().replace("-", "_").replace(" ", "_")
    return RELATION_ALIASES.get(p, p)


def _is_valid_entity_name(name: str) -> bool:
    stripped = name.strip()
    if not stripped or len(stripped) < 2:
        return False
    if stripped.lower() in INVALID_ENTITY_NAMES:
        return False
    if re.match(r'^[\d\s.,:/-]+$', stripped):
        return False
    if "'s " in stripped:
        return False
    if len(stripped.split()) > 3:
        return False
    if stripped.lower().startswith(("the ", "a ", "an ", "my ", "our ", "your ")):
        return False
    return True


def _resolve_chained_predicate(rel1: str, rel2: str) -> str:
    """
    Resolve chained possessives into a single predicate.
    "wife's father" → "father_in_law"
    "husband's mother" → "mother_in_law"
    "wife's brother" → "brother_in_law"
    If no mapping exists, return rel2 (the direct relation).
    """
    # spouse's parent = in-law
    spouse_rels = {"wife", "husband", "partner"}
    if rel1 in spouse_rels:
        if rel2 == "father":
            return "father_in_law"
        if rel2 == "mother":
            return "mother_in_law"
        if rel2 == "brother":
            return "brother_in_law"
        if rel2 == "sister":
            return "sister_in_law"

    # child's spouse = in-law
    child_rels = {"son", "daughter"}
    if rel1 in child_rels:
        if rel2 in ("wife", "husband"):
            return f"{rel1}_in_law"

    # parent's spouse
    if rel1 in ("father", "mother"):
        if rel2 in ("wife", "husband"):
            return rel1  # father's wife = mother (or step-mother), keep as parent

    # Default: just use the second relation
    return _normalize_predicate(rel2)


def _cap(s):
    """Auto-capitalize each word in a name."""
    if not s:
        return s
    return " ".join(w.capitalize() for w in s.strip().split())


# ── Relationship patterns ────────────────────────────────────
# These detect relationship PREDICATES. Entity names come from spaCy
# or from positional capture as fallback.

_INLAW_RELS = (
    r"(?i:father[\s-]in[\s-]law|mother[\s-]in[\s-]law|"
    r"brother[\s-]in[\s-]law|sister[\s-]in[\s-]law|"
    r"son[\s-]in[\s-]law|daughter[\s-]in[\s-]law|"
    r"step[\s-]?father|step[\s-]?mother)"
)

_SIMPLE_RELS = (
    r"(?i:father|mother|wife|husband|partner|brother|sister|"
    r"son|daughter|grandfather|grandmother|uncle|aunt|cousin|"
    r"boss|manager|friend|colleague|neighbor|classmate)"
)

# Patterns that extract (subject_indicator, predicate, object_indicator)
# "user" means the user entity; anything else is a name to resolve
RELATION_PATTERNS = [
    # "my father in law is X" / "my father-in-law is X"
    (r"(?i:my|our)\s+(" + _INLAW_RELS + r")\s+(?i:is|was|named)\s+(\w{3,})",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "X is my father in law"
    (r"(\w{3,})\s+(?i:is)\s+(?i:my|our)\s+(" + _INLAW_RELS + r")",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "X who is my father in law"
    (r"(\w{3,})\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _INLAW_RELS + r")",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "my father in law X talked/called..."
    (r"(?i:my|our)\s+(" + _INLAW_RELS + r")\s+(\w{3,})\b",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my wife is X" / "my father is X"
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was|named)\s+(\w{3,})",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "X is my wife" / "X is my father"
    (r"(\w{3,})\s+(?i:is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?!\s*[\s-](?i:in)[\s-](?i:law))",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "X who is my wifes father" — chained possessive
    # "wife's father" = father_in_law, "husband's mother" = mother_in_law, etc.
    (r"(\w{3,})\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?:'?s?)\s+(" + _SIMPLE_RELS + r")",
     lambda m: ("user",
                _resolve_chained_predicate(m.group(2).lower(), m.group(3).lower()),
                _cap(m.group(1)))),

    # "X who is my father" — but NOT "X who is my wifes father" (handled by chained pattern above)
    (r"(\w{3,})\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?!'?s?\s+(?i:father|mother|wife|husband|brother|sister))\b",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "my wife X" / "my father X" — name right after relation
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")\s+(\w{3,})\b",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my father's name is X"
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")'?s?\s+(?i:name)\s+(?i:is)\s+(\w{3,})",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "User's father is X" (compressed memory)
    (r"[Uu]ser'?s?\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was)\s+(?i:named?\s+)?(\w{3,})",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my wife's father is X" — chained possessive → resolve to in-law
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?:'s|s)\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was)\s+(\w{3,})",
     lambda m: ("user",
                _resolve_chained_predicate(m.group(1).lower(), m.group(2).lower()),
                _cap(m.group(3)))),

    # Pets
    (r"(?i:my|our)\s+(?i:dog|cat|pet)\s+(?i:is\s+)?(?i:named?\s+)?(\w{3,})",
     lambda m: ("user", "pet", _cap(m.group(1)))),
    (r"(?i:I|we)\s+(?i:have)\s+(?i:a|an)\s+(?i:dog|cat|pet)\s+(?i:named?|called)\s+(\w{3,})",
     lambda m: ("user", "pet", _cap(m.group(1)))),

    # Work/location
    (r"(?i:I|user)\s+(?i:work|works|worked)\s+(?i:at|for)\s+(\w{3,})",
     lambda m: ("user", "works_at", _cap(m.group(1)))),
    (r"(?i:I|we|user)\s+(?i:live|lives|lived|am\s+based|stay|stays)\s+(?i:in|at)\s+(\w{3,})",
     lambda m: ("user", "lives_in", _cap(m.group(1)))),

    # Name
    (r"(?i:my|our)\s+(?i:name)\s+(?i:is)\s+(\w{3,})",
     lambda m: ("user", "name", _cap(m.group(1)))),
    (r"(?i:I'm|I\s+am)\s+(\w{3,})\s*[.,!]?\s*$",
     lambda m: ("user", "name", _cap(m.group(1)))),
]


class EntityExtractor:
    """
    Three-tier entity and relationship extraction:
    1. spaCy NER — finds entity names from natural language
    2. Regex — detects relationship predicates
    3. Combines: spaCy entities + regex predicates = graph triples
    """

    def __init__(self, graph: GraphStore, user_name: str = "user", llm_fn=None):
        self.graph = graph
        self.user_name = user_name
        self.llm_fn = llm_fn
        self.nlp = _get_nlp()

        # Ensure the user entity exists
        existing = graph.find_entity_by_name(user_name)
        if existing:
            self.user_entity = existing
        else:
            self.user_entity = graph.add_entity(Entity(
                name=user_name, entity_type="person",
            ))

    def extract_from_memory(self, memory_content: str, memory_id: str = "") -> dict:
        """
        Extract entities and relations from text.

        Pipeline:
        1. spaCy NER finds named entities (people, places, orgs)
        2. Regex patterns find relationship predicates
        3. spaCy entities validate/enrich regex captures
        4. Combined results stored in graph
        """
        # Phase 1: spaCy NER — find all entity mentions
        spacy_entities = self._extract_spacy(memory_content)

        # Phase 2: Regex — find relationships, using spaCy to validate names
        result = self._extract_with_spacy_validation(
            memory_content, memory_id, spacy_entities
        )

        # Phase 3: Register any spaCy entities not captured by regex
        # (standalone mentions like "had lunch with Kamal near Galle")
        self._register_orphan_entities(spacy_entities, result, memory_id)

        return result

    # ── Phase 1: spaCy NER ────────────────────────────────────

    def _extract_spacy(self, text: str) -> dict[str, str]:
        """
        Run spaCy NER and return {name: entity_type} dict.
        Returns empty dict if spaCy unavailable.
        """
        if not self.nlp:
            return {}

        entities = {}
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in SPACY_SKIP_LABELS:
                    continue
                name = ent.text.strip()
                if not _is_valid_entity_name(name):
                    continue
                etype = SPACY_TYPE_MAP.get(ent.label_, "unknown")
                # spaCy sometimes misclassifies names (Prabhashi as ORG)
                # If it's a single word and not clearly an org, default to person
                if etype in ("company", "unknown") and len(name.split()) == 1:
                    etype = "person"
                entities[_cap(name)] = etype
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")

        return entities

    # ── Phase 2: Regex + spaCy validation ─────────────────────

    def _extract_with_spacy_validation(
        self, text: str, memory_id: str, spacy_entities: dict
    ) -> dict:
        """
        Run regex patterns. Use spaCy entities to validate extracted names.
        A name is valid if:
        - It was found by spaCy, OR
        - It passes the _is_valid_entity_name check
        """
        entities = []
        relations = []
        seen_relations = set()  # Deduplicate

        spacy_names_lower = {n.lower() for n in spacy_entities}

        for pattern, extractor in RELATION_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    result = extractor(match)
                except Exception:
                    continue

                if result is None:
                    continue
                if isinstance(result, tuple):
                    result = [result]

                for item in result:
                    if item is None:
                        continue
                    subj_name, predicate, obj_name = item
                    predicate = _normalize_predicate(predicate)

                    if subj_name == "user":
                        subj_name = self.user_name

                    # Validate the object name using three checks:
                    # 1. Is it a known spaCy entity? (most reliable)
                    # 2. Does it pass our name validation?
                    # 3. Is it NOT a common English word?
                    is_spacy_known = obj_name.lower() in spacy_names_lower
                    is_valid = _is_valid_entity_name(obj_name)

                    if not is_spacy_known and not is_valid:
                        continue

                    # Skip "name" predicate — handled by onboarding
                    if predicate == "name":
                        continue

                    # Deduplicate
                    rel_key = (subj_name.lower(), predicate, obj_name.lower())
                    if rel_key in seen_relations:
                        continue
                    seen_relations.add(rel_key)

                    # Resolve chained possessive subjects
                    if subj_name.lower() in VALID_PREDICATES and subj_name[0].islower():
                        resolved = self._resolve_relation_subject(subj_name)
                        if resolved:
                            subj_name = resolved

                    # Get entity type from spaCy if available
                    obj_type = spacy_entities.get(obj_name, None)

                    # Infer type from predicate if spaCy didn't find it
                    if not obj_type:
                        if predicate in ("works_at",):
                            obj_type = "company"
                        elif predicate in ("lives_in", "based_in", "from"):
                            obj_type = "place"
                        elif predicate in ("pet", "dog", "cat"):
                            obj_type = "animal"
                        elif predicate in ("interested_in", "hobby", "studies"):
                            obj_type = "concept"
                        else:
                            obj_type = "person"

                    # Create/find entities and relation
                    subj = self.graph.find_entity_by_name(subj_name)
                    if not subj:
                        subj = self.graph.add_entity(Entity(
                            name=subj_name, entity_type="person",
                            source_memory_id=memory_id,
                        ))
                        if not subj:
                            continue  # Entity rejected as junk
                        entities.append(subj)

                    obj = self.graph.find_entity_by_name(obj_name)
                    if not obj:
                        obj = self.graph.add_entity(Entity(
                            name=obj_name, entity_type=obj_type,
                            source_memory_id=memory_id,
                        ))
                        if not obj:
                            continue  # Entity rejected as junk
                        entities.append(obj)

                    rel = self.graph.add_relation(Relation(
                        subject_id=subj.id, predicate=predicate,
                        object_id=obj.id, source_memory_id=memory_id,
                    ))
                    relations.append(rel)

        return {"entities": entities, "relations": relations}

    def _resolve_relation_subject(self, relation_name: str) -> Optional[str]:
        """Resolve 'wife' → 'Prabhashi' by looking up user's relation."""
        user_entity = self.graph.find_entity_by_name(self.user_name)
        if not user_entity:
            return None
        row = self.graph.db.execute(
            "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
            (user_entity.id, relation_name.lower()),
        ).fetchone()
        if row:
            resolved = self.graph.db.execute(
                "SELECT name FROM entities WHERE id=?", (row[0],),
            ).fetchone()
            if resolved:
                return resolved[0]
        return None

    # ── Phase 3: Register orphan spaCy entities ───────────────

    def _register_orphan_entities(
        self, spacy_entities: dict, regex_result: dict, memory_id: str
    ):
        """
        Register spaCy entities that weren't captured by any regex pattern.
        These are standalone mentions (e.g., "had lunch with Kamal near Galle").
        """
        # Names already in the graph from regex extraction
        regex_names = set()
        for ent in regex_result.get("entities", []):
            regex_names.add(ent.name.lower())

        for name, etype in spacy_entities.items():
            if name.lower() in regex_names:
                continue
            if name.lower() == self.user_name.lower():
                continue
            # Only register if not already in graph
            existing = self.graph.find_entity_by_name(name)
            if not existing:
                entity = self.graph.add_entity(Entity(
                    name=name, entity_type=etype,
                    source_memory_id=memory_id,
                ))
                if entity:
                    regex_result["entities"].append(entity)
                    logger.debug(f"Registered orphan spaCy entity: {name} ({etype})")

    # ── LLM extraction (batch/deferred use only) ──────────────

    def extract_with_llm(self, text: str, memory_id: str = "") -> dict:
        """
        LLM-powered extraction. Call explicitly for batch processing
        (e.g., end_session compression), NOT during real-time observe().
        """
        if not self.llm_fn:
            return {"entities": [], "relations": []}

        prompt = (
            f"Extract ONLY proper noun entities (people, companies, places) and "
            f"their relationships from this text. The user's name is {self.user_name}.\n\n"
            f'Text: "{text}"\n\n'
            "Rules:\n"
            "- Entity names must be proper nouns (e.g., Prabhashi, Google, Singapore)\n"
            "- Do NOT create entities for common nouns or descriptions\n"
            "- Predicates must be from: father, mother, wife, husband, brother, sister, "
            "son, daughter, father_in_law, mother_in_law, works_at, lives_in, friend, pet\n\n"
            "Output format (one per line, nothing else):\n"
            "ENTITY: Name | type\n"
            "RELATION: Subject | predicate | Object\n\n"
            "If no entities found, respond: NONE"
        )

        try:
            result = self.llm_fn(prompt)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return {"entities": [], "relations": []}

        if "NONE" in result.upper()[:20]:
            return {"entities": [], "relations": []}

        entities = []
        relations = []

        for line in result.strip().split("\n"):
            line = line.strip()

            if line.startswith("ENTITY:"):
                parts = line[7:].strip().split("|")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    etype = parts[1].strip().lower()
                    if not _is_valid_entity_name(name):
                        continue
                    if etype not in ("person", "company", "place", "animal", "concept"):
                        etype = "unknown"
                    entity = self.graph.add_entity(Entity(
                        name=name, entity_type=etype,
                        source_memory_id=memory_id,
                    ))
                    entities.append(entity)

            elif line.startswith("RELATION:"):
                parts = line[9:].strip().split("|")
                if len(parts) >= 3:
                    subj_name = parts[0].strip()
                    predicate = _normalize_predicate(parts[1].strip())
                    obj_name = parts[2].strip()
                    if predicate not in VALID_PREDICATES:
                        continue
                    if not _is_valid_entity_name(obj_name):
                        continue

                    if subj_name.lower() in ("user", self.user_name.lower()):
                        subj_name = self.user_name

                    subj = self.graph.find_entity_by_name(subj_name)
                    if not subj:
                        subj = self.graph.add_entity(Entity(name=subj_name, entity_type="person"))
                    obj = self.graph.find_entity_by_name(obj_name)
                    if not obj:
                        obj = self.graph.add_entity(Entity(name=obj_name))

                    rel = self.graph.add_relation(Relation(
                        subject_id=subj.id, predicate=predicate,
                        object_id=obj.id, source_memory_id=memory_id,
                    ))
                    relations.append(rel)

        return {"entities": entities, "relations": relations}
