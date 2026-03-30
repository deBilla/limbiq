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
    # Attributes / conditions
    "has_condition", "is_a", "diagnosed_with",
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

# Static junk — LLM artifacts, pronouns, stop words
_INVALID_ENTITY_BASE = {
    # LLM artifacts
    "none", "null", "n/a", "unknown", "undefined", "default", "",
    "?", "user", "the user", "assistant", "ai", "bot",
    # Common verbs / participles
    "has", "have", "had", "was", "were", "is", "are", "been", "be", "am",
    "do", "does", "did", "done", "doing",
    "say", "says", "said", "saying",
    "go", "goes", "went", "gone", "going",
    "get", "gets", "got", "getting",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "known", "knowing",
    "think", "thinks", "thought", "thinking",
    "take", "takes", "took", "taken", "taking",
    "see", "sees", "saw", "seen", "seeing",
    "come", "comes", "came", "coming",
    "want", "wants", "wanted", "wanting",
    "look", "looks", "looked", "looking",
    "use", "uses", "used", "using",
    "find", "finds", "found", "finding",
    "give", "gives", "gave", "given", "giving",
    "tell", "tells", "told", "telling",
    "work", "works", "worked", "working",
    "call", "calls", "called", "calling",
    "try", "tries", "tried", "trying",
    "ask", "asks", "asked", "asking",
    "need", "needs", "needed", "needing",
    "feel", "feels", "felt", "feeling",
    "become", "becomes", "became", "becoming",
    "leave", "leaves", "left", "leaving",
    "put", "puts", "putting",
    "mean", "means", "meant", "meaning",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "letting",
    "begin", "begins", "began", "beginning",
    "show", "shows", "showed", "shown", "showing",
    "hear", "hears", "heard", "hearing",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "named", "called", "mentioned", "based",
    "being", "having",
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # Demonstratives / determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Conjunctions / prepositions / adverbs
    "and", "but", "or", "nor", "for", "yet", "so",
    "if", "then", "else", "because", "since", "when", "while", "although",
    "after", "before", "until", "unless", "though", "whether",
    "in", "on", "at", "to", "of", "by", "up", "out", "off", "into",
    "not", "also", "just", "very", "really", "actually", "already",
    "still", "even", "only", "now", "here", "there", "then",
    "about", "with", "from", "will", "would", "could", "should", "shall",
    "may", "might", "can", "must",
    "other", "another", "some", "any", "all", "each", "every",
    "many", "much", "more", "most", "few", "less", "least",
    # Common non-entity words that get capitalized at sentence starts
    "yes", "no", "ok", "okay", "sure", "well", "oh", "hi", "hello", "hey",
    "thanks", "thank", "please", "sorry",
    "new", "old", "big", "small", "good", "bad", "great",
    "home", "school", "office", "hospital", "doctor",
    "condition", "schedule", "care", "feeding", "well-being",
    "advice", "help", "question", "answer", "problem", "issue",
    "who", "what", "where", "when", "how", "which", "whom", "whose", "why",
    # Common sentence-start words that are NOT entities
    "however", "therefore", "furthermore", "moreover", "meanwhile",
    "sometimes", "always", "never", "often", "usually", "perhaps",
    "maybe", "probably", "certainly", "definitely", "apparently",
    "today", "tomorrow", "yesterday", "recently", "currently",
    "first", "second", "third", "last", "next", "finally",
    "everything", "something", "nothing", "anything", "everyone",
    "someone", "anyone", "no one", "nobody",
    # LLM response artifacts — words that start LLM responses
    "understood", "noted", "confirmed", "acknowledged", "specified",
    "correct", "right", "absolutely", "exactly", "indeed",
    "following", "key", "point", "points", "example", "examples",
    "summary", "information", "details", "context", "note",
    "important", "remember", "recall", "mention", "mentioned",
    "regarding", "concerning", "related", "specifically",
    "happy", "glad", "sorry", "appreciate", "apologies",
}

# Derived dynamically: any word that is a valid predicate or alias is a
# relationship descriptor, not an entity name. This means adding a new
# predicate to VALID_PREDICATES or RELATION_ALIASES automatically prevents
# it from being stored as an entity — no hardcoding needed.
INVALID_ENTITY_NAMES = (
    _INVALID_ENTITY_BASE
    | {p.replace("_", " ") for p in VALID_PREDICATES}
    | {p.replace("_", "-") for p in VALID_PREDICATES}
    | VALID_PREDICATES
    | set(RELATION_ALIASES.keys())
)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_match_predicate(word: str) -> str | None:
    """Find the closest valid predicate within Levenshtein distance 1.

    Only attempts fuzzy matching for words with length >= 4 to avoid
    false positives on short words (e.g. "pet" matching "set").
    Returns the matched predicate or None if no close match.
    """
    if len(word) < 4:
        return None

    candidates = list(VALID_PREDICATES) + list(RELATION_ALIASES.keys())
    best_match = None
    best_dist = 2  # Only accept distance <= 1

    for candidate in candidates:
        # Skip candidates with very different lengths (can't be dist ≤ 1)
        if abs(len(candidate) - len(word)) > 1:
            continue
        dist = _levenshtein_distance(word, candidate)
        if dist < best_dist:
            best_dist = dist
            best_match = candidate

    if best_match is None:
        return None

    # Resolve through aliases if the match was an alias key
    resolved = RELATION_ALIASES.get(best_match, best_match)
    logger.info(f"Fuzzy predicate match: '{word}' → '{resolved}' (distance={best_dist})")
    return resolved


def _normalize_predicate(pred: str) -> str:
    p = pred.lower().strip().replace("-", "_").replace(" ", "_")
    result = RELATION_ALIASES.get(p, p)
    # If exact match found in VALID_PREDICATES or aliases, return it
    if result in VALID_PREDICATES:
        return result
    # Try fuzzy matching for typos (distance ≤ 1)
    fuzzy = _fuzzy_match_predicate(p)
    if fuzzy:
        return fuzzy
    return result


def _is_valid_entity_name(name: str) -> bool:
    stripped = name.strip()
    # Strip possessives first
    stripped = _strip_possessive(stripped)
    # Strip parenthetical content: "User (Dimuthu)" → "User"
    stripped = re.sub(r'\s*\([^)]*\)\s*$', '', stripped).strip()
    if not stripped or len(stripped) < 2:
        return False
    if stripped.lower() in INVALID_ENTITY_NAMES:
        return False
    if re.match(r'^[\d\s.,:/-]+$', stripped):
        return False
    if "'s " in stripped or "\u2019s " in stripped:
        return False
    if len(stripped.split()) > 3:
        return False
    if stripped.lower().startswith(("the ", "a ", "an ", "my ", "our ", "your ")):
        return False

    words = stripped.split()

    # Reject multi-word names starting with sentence connectives/particles
    # ("If User", "No Renuka", "Thank User", "But Prabhashi")
    if len(words) >= 2:
        _SENTENCE_STARTERS = {
            "if", "no", "yes", "ok", "okay", "so", "and", "but", "or",
            "because", "since", "while", "when", "where", "what", "who",
            "which", "how", "why", "therefore", "however", "actually",
            "really", "just", "then", "also", "moreover", "furthermore",
            "thank", "thanks", "please", "hi", "hello", "hey",
            "not", "nor", "yet", "for", "do", "does", "did",
        }
        if words[0].lower() in _SENTENCE_STARTERS:
            return False

    # Reject verb-noun fragments that snuck past greedy regex captures
    # ("Dexter Has Megaesophagus", "John Works At")
    _FRAGMENT_VERBS = {"has", "had", "have", "is", "was", "were", "does", "did",
                       "works", "lives", "goes", "says", "gets", "takes", "makes"}
    if len(words) >= 2 and words[1].lower() in _FRAGMENT_VERBS:
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


def _strip_possessive(s: str) -> str:
    """Strip trailing possessive forms: 's, 's, s (on proper names)."""
    s = s.strip()
    # Explicit possessive with apostrophe
    if s.endswith(("\u2019s", "'s")):
        return s[:-2].strip()
    return s


def _cap(s):
    """Auto-capitalize each word in a name, strip possessives."""
    if not s:
        return s
    s = _strip_possessive(s.strip())
    return " ".join(w.capitalize() for w in s.split())


# ── Relationship patterns ────────────────────────────────────
# These detect relationship PREDICATES. Entity names come from spaCy
# or from positional capture as fallback.

_INLAW_RELS = (
    r"(?i:father[\s-]in[\s-]law|mother[\s-]in[\s-]law|"
    r"brother[\s-]in[\s-]law|sister[\s-]in[\s-]law|"
    r"son[\s-]in[\s-]law|daughter[\s-]in[\s-]law|"
    r"step[\s-]?father|step[\s-]?mother)"
)

# Build from VALID_PREDICATES + RELATION_ALIASES so adding a new
# predicate or alias automatically makes it matchable in regex patterns.
_SIMPLE_REL_WORDS = sorted(
    {p.replace("_", " ") for p in VALID_PREDICATES
     if "_in_law" not in p and "step_" not in p and p not in (
         "works_at", "role", "lives_in", "based_in", "from",
         "interested_in", "studies", "hobby", "has_condition",
         "is_a", "diagnosed_with", "married_to", "parent",
         "sibling", "child", "pet",
     )}
    | set(RELATION_ALIASES.keys()),
    key=len, reverse=True,  # longest first to avoid partial matches
)
_SIMPLE_RELS = r"(?i:" + "|".join(re.escape(w) for w in _SIMPLE_REL_WORDS) + r")"

# Patterns that extract (subject_indicator, predicate, object_indicator)
# "user" means the user entity; anything else is a name to resolve
RELATION_PATTERNS = [
    # "my father in law is X" / "my father-in-law is X"
    (r"(?i:my|our)\s+(" + _INLAW_RELS + r")\s+(?i:is|was|named)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "X is my father in law"
    (r"([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\s+(?i:is)\s+(?i:my|our)\s+(" + _INLAW_RELS + r")",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "X who is my father in law"
    (r"([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _INLAW_RELS + r")",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "my father in law X talked/called..."
    (r"(?i:my|our)\s+(" + _INLAW_RELS + r")\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\b",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my wife is X" / "my father is X"
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was|named)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "X is my wife" / "X is my father"
    (r"([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\s+(?i:is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?!\s*[\s-](?i:in)[\s-](?i:law))",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "X who is my wifes father" — chained possessive
    # "wife's father" = father_in_law, "husband's mother" = mother_in_law, etc.
    (r"([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?:'?s?)\s+(" + _SIMPLE_RELS + r")",
     lambda m: ("user",
                _resolve_chained_predicate(m.group(2).lower(), m.group(3).lower()),
                _cap(m.group(1)))),

    # "X who is my father" — but NOT "X who is my wifes father" (handled by chained pattern above)
    (r"([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)\s+(?i:who\s+is)\s+(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?!'?s?\s+(?i:father|mother|wife|husband|brother|sister))\b",
     lambda m: ("user", _normalize_predicate(m.group(2)), _cap(m.group(1)))),

    # "my wife X" / "my father X" — name right after relation
    # Name stops at common verbs/prepositions to avoid eating the rest of the sentence
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)(?:\s+(?:is|was|has|had|who|and|or|in|at|from|to|the|a|an|recently|also|just|then|called|said|went|told|asked|lives?|works?)\b|[.,;!?]|\s*$)",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my father's name is X"
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")'?s?\s+(?i:name)\s+(?i:is)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "User's father is X" (compressed memory)
    (r"[Uu]ser'?s?\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was)\s+(?i:named?\s+)?([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user", _normalize_predicate(m.group(1)), _cap(m.group(2)))),

    # "my wife's father is X" — chained possessive → resolve to in-law
    (r"(?i:my|our)\s+(" + _SIMPLE_RELS + r")(?:'s|s)\s+(" + _SIMPLE_RELS + r")\s+(?i:is|was)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user",
                _resolve_chained_predicate(m.group(1).lower(), m.group(2).lower()),
                _cap(m.group(3)))),

    # Pets — sentence-boundary lookahead stops greedy capture
    # ("my dog Dexter has megaesophagus" → captures "Dexter", NOT "Dexter Has Megaesophagus")
    (r"(?i:my|our)\s+(?i:dog|cat|pet)\s+(?i:is\s+)?(?i:named?\s+)?([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)(?=\s+(?:has|is|was|had|who|and|but|which|that|with|the|a|an|recently|also|just|then)\b|[.,;!?]|\s*$)",
     lambda m: ("user", "pet", _cap(m.group(1)))),
    (r"(?i:I|we)\s+(?i:have)\s+(?i:a|an)\s+(?i:dog|cat|pet)\s+(?i:named?|called)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)(?=\s+(?:has|is|was|had|who|and|but|which|that|with|the|a|an|recently|also|just|then)\b|[.,;!?]|\s*$)",
     lambda m: ("user", "pet", _cap(m.group(1)))),

    # Work/location — same boundary stops
    (r"(?i:I|user)\s+(?i:work|works|worked)\s+(?i:at|for)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)(?=\s+(?:as|in|and|but|since|for|from|doing|where|which|the|a|an)\b|[.,;!?]|\s*$)",
     lambda m: ("user", "works_at", _cap(m.group(1)))),
    (r"(?i:I|we|user)\s+(?i:live|lives|lived|am\s+based|stay|stays)\s+(?i:in|at)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)?)(?=\s+(?:and|but|with|since|for|near|which|the|a|an)\b|[.,;!?]|\s*$)",
     lambda m: ("user", "lives_in", _cap(m.group(1)))),

    # Name
    (r"(?i:my|our)\s+(?i:name)\s+(?i:is)\s+([A-Za-z][a-zA-Z]{2,}(?:\s+[A-Za-z][a-zA-Z]+)*)",
     lambda m: ("user", "name", _cap(m.group(1)))),
    (r"(?:I'm|I\s+am)\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*)\s*[.,!]?\s*$",
     lambda m: ("user", "name", _cap(m.group(1)))),

    # ── Entity-to-entity relationships (not involving user) ───

    # "X's father is Y" / "Xs father is Y" — third-person possessive
    # Handles both "Prabhashi's father" and "Prabhashis father" (without apostrophe)
    (r"([A-Z][a-zA-Z]{2,})(?:'s|s)\s+(" + _SIMPLE_RELS + r")\s+(?:is|was)\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*)",
     lambda m: (_cap(_strip_possessive(m.group(1))), _normalize_predicate(m.group(2)), _cap(m.group(3)))
     if _is_valid_entity_name(_strip_possessive(m.group(1))) and _is_valid_entity_name(m.group(3)) else None),

    # "X's father Y" — third-person possessive without "is"
    (r"([A-Z][a-zA-Z]{2,})(?:'s|s)\s+(" + _SIMPLE_RELS + r")\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)(?=\s+(?:is|was|has|had|who|and|or|in|at|from|to|the|a|an|recently|also|just|then|called|said|went|told|asked)\b|[.,;!?]|\s*$)",
     lambda m: (_cap(_strip_possessive(m.group(1))), _normalize_predicate(m.group(2)), _cap(m.group(3)))
     if _is_valid_entity_name(_strip_possessive(m.group(1))) and _is_valid_entity_name(m.group(3)) else None),

    # "X has Y" / "X has a condition called Y" — medical/attribute
    (r"(\w{3,})\s+(?i:has|have|had)\s+(?i:a\s+(?:condition|disease|illness)\s+(?:called|named)\s+)?(\w{3,})",
     lambda m: (_cap(m.group(1)), "has_condition", m.group(2).lower())
     if _is_valid_entity_name(m.group(1)) and _is_valid_entity_name(m.group(2)) else None),

    # "X is a Y" / "X is an Y" — type/role (e.g., "Chandrasiri is a doctor")
    (r"([A-Z]\w{2,})\s+(?i:is|was)\s+(?i:a|an)\s+(\w+(?:\s+\w+)?)",
     lambda m: (_cap(m.group(1)), "is_a", m.group(2).strip().lower())
     if _is_valid_entity_name(m.group(1)) else None),

    # "X lives in Y" — third-person location (case-insensitive for both names)
    (r"([A-Za-z]\w{2,})\s+(?i:lives?|lived|stay|stays)\s+(?i:in|at)\s+([A-Za-z]\w{2,}(?:\s+[A-Za-z]\w+)?)",
     lambda m: (_cap(m.group(1)), "lives_in", _cap(m.group(2)))
     if _is_valid_entity_name(m.group(1)) else None),

    # "X works at Y" — third-person work (case-insensitive for both names)
    (r"([A-Za-z]\w{2,})\s+(?i:works?|worked)\s+(?i:at|for)\s+([A-Za-z]\w{2,})",
     lambda m: (_cap(m.group(1)), "works_at", _cap(m.group(2)))
     if _is_valid_entity_name(m.group(1)) else None),
]


class EntityExtractor:
    """
    Four-tier entity and relationship extraction:
    1. Transformer encoder — produces embeddings for entity spans (shared space with memories)
    2. spaCy NER — finds entity names from natural language
    3. Regex — detects relationship predicates
    4. Combines: encoder embeddings + spaCy entities + regex predicates = graph triples

    The transformer encoder runs FIRST, producing embeddings that are stored
    with each entity node in the graph. This means entity nodes start life
    with semantically meaningful representations in the SAME vector space
    as memory embeddings — enabling similarity-based graph operations.
    """

    def __init__(self, graph: GraphStore, user_name: str = "user", llm_fn=None,
                 embedding_engine=None):
        self.graph = graph
        self.user_name = user_name
        self.llm_fn = llm_fn
        self.nlp = _get_nlp()
        self._pending_extractions: list[tuple[str, str]] = []
        self._embedding_engine = embedding_engine

        # Initialize transformer encoder if embedding engine available
        self._encoder = None
        if embedding_engine is not None:
            try:
                from limbiq.graph.encoder import TransformerEntityEncoder
                self._encoder = TransformerEntityEncoder(embedding_engine)
                logger.info("Transformer entity encoder initialized")
            except Exception as e:
                logger.warning(f"Could not initialize transformer encoder: {e}")

        # Ensure the user entity exists
        existing = graph.find_entity_by_name(user_name)
        if existing:
            self.user_entity = existing
        else:
            self.user_entity = graph.add_entity(Entity(
                name=user_name, entity_type="person",
            ))

    def extract_from_memory(
        self, memory_content: str, memory_id: str = "",
        response_mode: bool = False,
    ) -> dict:
        """
        Extract entities and relations from text.

        Args:
            memory_content: Text to extract from.
            memory_id: Optional memory ID for provenance.
            response_mode: If True, only extract relations between entities
                that already exist in the graph. This prevents LLM response
                artifacts ("stay in touch", "quite serious") from polluting
                the graph while still capturing confirmed relationships
                ("your father-in-law Chandrasiri").

        Hybrid pipeline:
        0. Transformer encoder pass (produces embeddings for all extracted entities)
        1. Tier 1: spaCy dependency parsing (handles ~90% of standard English)
           Fallback: regex patterns (when spaCy is unavailable)
        2. Tier 2: LLM tie-break (for uncertain/ambiguous fragments from Tier 1)
        3. Register orphan spaCy NER entities (names without detected relations)
        4. Deep LLM extraction (deferred batch, background)
        """
        # Phase 0: Transformer encoder pass — produces entity embeddings
        # and optional learned entity/relation detection
        encoder_output = None
        if not response_mode and self._encoder is not None:
            try:
                from limbiq.graph.encoder import EncoderOutput
                encoder_output = self._encoder.encode(memory_content, self.user_name)
            except Exception as e:
                logger.warning(f"Encoder pass failed, continuing: {e}")

        # Phase 1: Structural extraction
        uncertain = []
        if self.nlp:
            # Tier 1: Dependency-based extraction (primary)
            # In response_mode, same pipeline runs but entity creation is
            # gated — only entities already in the graph get new relations.
            result, uncertain = self._extract_from_dependencies(
                memory_content, memory_id, create_new_entities=not response_mode,
            )
        else:
            # Fallback: Regex-based extraction (when spaCy unavailable)
            spacy_entities = {}
            result = self._extract_with_spacy_validation(
                memory_content, memory_id, spacy_entities
            )

        # Tier 2: LLM resolution for uncertain fragments
        if uncertain and self.llm_fn and not response_mode:
            self._resolve_uncertain_with_llm(uncertain, result, memory_id)

        # Register orphan spaCy NER entities (names without relations)
        if not response_mode:
            spacy_entities = self._extract_spacy(memory_content)
            self._register_orphan_entities(spacy_entities, result, memory_id)

        # Merge encoder output — attach embeddings to extracted entities
        if encoder_output is not None:
            self._merge_encoder_output(encoder_output, result, memory_id)

        # Queue for deep LLM extraction (runs in background)
        if self.llm_fn and len(memory_content.strip()) > 20 and not response_mode:
            self._pending_extractions.append((memory_content, memory_id))
            logger.info(f"Queued for LLM extraction ({len(self._pending_extractions)} pending): {memory_content[:60]}...")

        return result

    def _merge_encoder_output(self, encoder_output, regex_result: dict, memory_id: str):
        """Merge transformer encoder output with regex extraction results.

        1. Attach embeddings to entities already extracted by regex
        2. Add new entities detected by encoder but missed by regex
        3. Add new relations detected by encoder's relation classifier
        """
        # Build lookup of already-extracted entity names
        extracted_names = set()
        for ent in regex_result.get("entities", []):
            extracted_names.add(ent.name.lower())

        # Attach embeddings to existing entities in the graph
        for enc_ent in encoder_output.entities:
            existing = self.graph.find_entity_by_name(enc_ent.name)
            if existing and enc_ent.embedding:
                # Store embedding as entity property for graph operations
                try:
                    import json
                    props = existing.properties or {}
                    # Don't store full embedding in JSON — too large
                    # Instead, store a flag that embedding exists
                    props["has_encoder_embedding"] = True
                    self.graph.db.execute(
                        "UPDATE entities SET properties=? WHERE id=?",
                        (json.dumps(props), existing.id),
                    )
                    self.graph.db.commit()
                except Exception:
                    pass

            # Add encoder-detected entities that regex missed
            if enc_ent.name.lower() not in extracted_names:
                if _is_valid_entity_name(enc_ent.name):
                    entity = self.graph.add_entity(Entity(
                        name=enc_ent.name,
                        entity_type=enc_ent.entity_type,
                        source_memory_id=memory_id,
                    ))
                    if entity:
                        regex_result["entities"].append(entity)
                        extracted_names.add(enc_ent.name.lower())

        # Add encoder-detected relations (supplementary to regex)
        # Strategy: regex/spaCy results are trusted. Encoder only fills gaps.
        # Build set of existing relation triples for dedup
        existing_rels = set()
        for ent in regex_result.get("entities", []):
            for r in self.graph.get_relations_for(ent.id) if hasattr(ent, 'id') else []:
                existing_rels.add((r.subject_id, r.predicate, r.object_id))

        for enc_rel in encoder_output.relations:
            if enc_rel.confidence < 0.6:  # Higher threshold for encoder (supplementary)
                continue
            if enc_rel.predicate == "none":
                continue

            subj = self.graph.find_entity_by_name(enc_rel.subject.name)
            obj = self.graph.find_entity_by_name(enc_rel.object.name)
            if subj and obj:
                # Skip if regex already extracted a relation between these entities
                has_existing = any(
                    (s, p, o) for s, p, o in existing_rels
                    if s == subj.id and o == obj.id
                )
                if has_existing:
                    # Regex is trusted — don't override
                    if enc_rel.predicate not in {r[1] for r in existing_rels
                                                  if r[0] == subj.id and r[2] == obj.id}:
                        logger.info(
                            f"Encoder disagrees with regex: {enc_rel.subject.name}"
                            f"→{enc_rel.predicate}→{enc_rel.object.name} "
                            f"(keeping regex result)"
                        )
                    continue

                if enc_rel.predicate in VALID_PREDICATES:
                    self.graph.add_relation(Relation(
                        subject_id=subj.id,
                        predicate=enc_rel.predicate,
                        object_id=obj.id,
                        confidence=enc_rel.confidence,
                        source_memory_id=memory_id,
                    ))

    def train_encoder(self) -> dict:
        """Train the transformer encoder classifiers from existing graph data.

        Call this after the graph has accumulated enough entities/relations
        from regex+LLM extraction. The encoder learns to replicate and
        improve upon the existing extraction pipeline.
        """
        if self._encoder is None:
            return {"status": "no_encoder"}
        return self._encoder.train_from_graph(self.graph)

    def process_pending_extractions(self):
        """
        Run LLM extraction on queued messages. Call from background task.
        Processes all pending messages in ONE LLM call for efficiency.
        """
        if not self.llm_fn:
            logger.debug("No LLM function — skipping deep extraction")
            return {"entities": [], "relations": []}

        if not self._pending_extractions:
            logger.debug("No pending extractions")
            return {"entities": [], "relations": []}

        pending = list(self._pending_extractions)
        self._pending_extractions.clear()

        logger.info(f"Processing {len(pending)} pending extractions via LLM")

        # Batch all pending texts into one LLM call
        combined_text = "\n---\n".join(text for text, _ in pending)
        memory_id = pending[0][1] if pending else ""

        result = self.extract_with_llm(combined_text, memory_id)
        logger.info(f"LLM extraction result: {len(result.get('entities', []))} entities, {len(result.get('relations', []))} relations")
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

    # ── Tier 1: Dependency-based extraction ────────────────────

    def _extract_from_dependencies(self, text: str, memory_id: str,
                                    create_new_entities: bool = True) -> tuple[dict, list[str]]:
        """
        Tier 1: Extract entities and relations using spaCy dependency parsing.

        Walks the parse tree looking for three structural patterns:
        1. Possessive relations: "my wife Prabhashi" / "my wife's father Chandrasiri"
        2. Subject-verb-object: "my Dog has megaesophagus" / "I work at Google"
        3. Copular attribution: "Chandrasiri is a doctor"

        Returns:
            (result_dict, uncertain_snippets) — uncertain snippets are text
            fragments that had a possessive or verb structure but couldn't be
            fully resolved; these are passed to Tier 2 (LLM) for resolution.
        """
        doc = self.nlp(text)
        entities = []
        relations = []
        seen = set()
        uncertain = []

        # ── Pass 1: First-person possessive relations ──
        poss_results = self._dep_possessive_relations(doc, memory_id)
        for subj_name, predicate, obj_name, obj_type, confidence in poss_results:
            rel_key = (subj_name.lower(), predicate, obj_name.lower())
            if rel_key in seen:
                continue
            seen.add(rel_key)

            subj = self.graph.find_entity_by_name(subj_name)
            if not subj:
                if not create_new_entities:
                    continue
                subj = self.graph.add_entity(Entity(
                    name=subj_name, entity_type="person",
                    source_memory_id=memory_id,
                ))
                if not subj:
                    continue
                entities.append(subj)

            obj = self.graph.find_entity_by_name(obj_name)
            if not obj:
                if not create_new_entities:
                    continue
                obj = self.graph.add_entity(Entity(
                    name=obj_name, entity_type=obj_type,
                    source_memory_id=memory_id,
                ))
                if not obj:
                    continue
                entities.append(obj)

            rel = self.graph.add_relation(Relation(
                subject_id=subj.id, predicate=predicate,
                object_id=obj.id, confidence=confidence,
                source_memory_id=memory_id,
            ))
            relations.append(rel)

        # ── Pet merge: when a named pet is added and a generic pet ──
        # ── ("Dog", "Cat") already exists, merge them. ──
        pet_words = {'dog', 'cat', 'pet'}
        for subj_name, predicate, obj_name, obj_type, _ in poss_results:
            if predicate != 'pet' or obj_name.lower() in pet_words:
                continue
            # This is a named pet (e.g., "Dexter") — check for generic pet to merge
            user_ent = self.graph.find_entity_by_name(subj_name)
            if not user_ent:
                continue
            pet_rels = self.graph.db.execute(
                "SELECT r.object_id, e.name FROM relations r "
                "JOIN entities e ON e.id = r.object_id "
                "WHERE r.subject_id=? AND r.predicate='pet' AND r.is_inferred=0",
                (user_ent.id,),
            ).fetchall()
            for old_pet_id, old_pet_name in pet_rels:
                if old_pet_name.lower() in pet_words and old_pet_name.lower() != obj_name.lower():
                    named_ent = self.graph.find_entity_by_name(obj_name)
                    if named_ent:
                        try:
                            # Re-point all relations from generic → named entity
                            self.graph.db.execute(
                                "UPDATE relations SET subject_id=? WHERE subject_id=?",
                                (named_ent.id, old_pet_id),
                            )
                            self.graph.db.execute(
                                "UPDATE relations SET object_id=? WHERE object_id=?",
                                (named_ent.id, old_pet_id),
                            )
                            # Delete duplicate pet relation (user→pet→generic)
                            self.graph.db.execute(
                                "DELETE FROM relations WHERE subject_id=? AND predicate='pet' AND object_id=?",
                                (user_ent.id, named_ent.id),
                            )
                            # Re-create single pet relation
                            self.graph.add_relation(Relation(
                                subject_id=user_ent.id, predicate="pet",
                                object_id=named_ent.id,
                            ))
                            # Delete the generic entity
                            self.graph.db.execute(
                                "DELETE FROM entities WHERE id=?", (old_pet_id,),
                            )
                            self.graph.db.commit()
                            logger.info(f"Pet merge: {old_pet_name} → {obj_name}")
                        except Exception as e:
                            logger.warning(f"Pet merge failed: {e}")

        # ── Pass 2: Subject-verb-object relations ──
        svo_results, svo_uncertain = self._dep_verb_relations(doc, memory_id, seen)
        for subj_name, predicate, obj_name, obj_type, confidence in svo_results:
            rel_key = (subj_name.lower(), predicate, obj_name.lower())
            if rel_key in seen:
                continue
            seen.add(rel_key)

            subj = self.graph.find_entity_by_name(subj_name)
            if not subj:
                continue  # Subject must already exist from Pass 1 or prior data

            obj = self.graph.find_entity_by_name(obj_name)
            if not obj:
                if not create_new_entities:
                    continue
                obj = self.graph.add_entity(Entity(
                    name=obj_name, entity_type=obj_type,
                    source_memory_id=memory_id,
                ))
                if not obj:
                    continue
                entities.append(obj)

            rel = self.graph.add_relation(Relation(
                subject_id=subj.id, predicate=predicate,
                object_id=obj.id, confidence=confidence,
                source_memory_id=memory_id,
            ))
            relations.append(rel)

        uncertain.extend(svo_uncertain)

        return {"entities": entities, "relations": relations}, uncertain

    def _dep_possessive_relations(self, doc, memory_id: str) -> list[tuple]:
        """
        Find all first-person possessive relations in the dependency tree.

        Pattern: my/our → (poss) → relation_noun → (appos/compound) → Name

        Handles:
        - Simple: "my wife Prabhashi" → (user, wife, Prabhashi, person, 0.95)
        - Chained: "my wifes father Chandrasiri" → (user, father_in_law, Chandrasiri, person, 0.95)
        - Generic pet: "my Dog" → (user, pet, Dog, animal, 0.9)

        Returns list of (subject_name, predicate, object_name, object_type, confidence).
        """
        results = []
        processed_heads = set()  # Avoid extracting same NP twice

        for token in doc:
            if token.dep_ != 'poss':
                continue

            # Accept first-person possessives and user-name possessives
            # ("my wife", "our dog", "User's father", "Dimuthu's wife")
            is_first_person = token.text.lower() in ('my', 'our')
            is_user_ref = token.text.lower() in (
                'user', self.user_name.lower()
            ) or token.text.lower().rstrip("'s") in (
                'user', self.user_name.lower()
            )
            # Also accept "your" (from LLM response: "your father-in-law")
            is_second_person = token.text.lower() in ('your', 'you')

            # Third-person possessives: resolve "her/his/their" to the
            # nearest preceding entity in the same sentence.
            # "My wife was ... with her father Chandrasiri"
            #   → "her" resolves to "wife" → chain: wife.father = father_in_law
            is_third_person_resolved = False
            resolved_antecedent = None
            if token.text.lower() in ('her', 'his', 'their'):
                resolved_antecedent = self._resolve_pronoun(token, doc)
                if resolved_antecedent:
                    is_third_person_resolved = True

            if not (is_first_person or is_user_ref or is_second_person
                    or is_third_person_resolved):
                continue

            head = token.head
            if head.i in processed_heads:
                continue

            # Check if the DIRECT head is a relation/pet word.
            # "my wife Prabhashi" → head=wife (relation word) → proceed
            # "my Dog has..." → head=Dog (pet word) → pet entity
            # "my dogs health" → head=health (not a relation) → skip
            head_norm = _normalize_predicate(head.lemma_.lower())
            head_is_relation = head_norm in VALID_PREDICATES or head.lemma_.lower() in RELATION_ALIASES

            # Special case: "My Dog dexter" → head=dexter (not a relation word),
            # but has a compound child "Dog" that IS a pet word.
            # Treat this as a named pet: pet → Dexter.
            if not head_is_relation:
                pet_compound = self._check_pet_compound(head)
                if pet_compound:
                    name = self._collect_name_span(head, doc)
                    if name and _is_valid_entity_name(name):
                        results.append((self.user_name, "pet", _cap(name), "animal", 0.95))
                        processed_heads.add(head.i)
                continue

            # Build the relation chain by walking UP through the tree
            # from the head of "my" toward the named entity
            chain, name_token = self._walk_possessive_tree(head, token, doc)
            if not chain:
                continue

            # Mark all chain tokens as processed
            for t in chain:
                processed_heads.add(t.i)

            # Resolve the chain into a single predicate.
            # If pronoun was resolved (e.g., "her" → "wife"), prepend the
            # antecedent's relation to form a chain: wife + father → father_in_law
            pred_words = [_normalize_predicate(t.lemma_.lower()) for t in chain]
            if is_third_person_resolved and resolved_antecedent:
                antecedent_pred = _normalize_predicate(resolved_antecedent)
                if antecedent_pred in VALID_PREDICATES:
                    pred_words.insert(0, antecedent_pred)

            valid_preds = [p for p in pred_words if p in VALID_PREDICATES]

            if not valid_preds:
                continue

            if len(valid_preds) == 1:
                predicate = valid_preds[0]
            else:
                predicate = _resolve_chained_predicate(valid_preds[0], valid_preds[-1])

            if predicate not in VALID_PREDICATES:
                continue

            # Normalize specific predicates: dog/cat → pet
            if predicate in ('dog', 'cat'):
                predicate = 'pet'

            # Find the proper noun name
            if name_token:
                name = self._collect_name_span(name_token, doc)
            else:
                # No name found — if this is a pet word, use the lemma
                # (singular form) as the entity name: "dogs" → "Dog"
                pet_words = {'dog', 'cat', 'pet'}
                if chain[-1].lemma_.lower() in pet_words:
                    name = chain[-1].lemma_.capitalize()
                    predicate = "pet"
                    results.append((self.user_name, predicate, name, "animal", 0.9))
                    continue
                else:
                    continue

            if not _is_valid_entity_name(name):
                continue

            obj_type = self._type_from_predicate(predicate)
            results.append((self.user_name, predicate, _cap(name), obj_type, 0.95))

        return results

    def _walk_possessive_tree(self, head, poss_token, doc):
        """
        Walk the dependency tree from the possessive head to find
        the chain of relation words and the terminal name.

        For "My wifes father Chandrasiri called":
          head=wifes → wifes is nsubj of father? No, father is compound of Chandrasiri.
          Actually: My→poss→wifes, wifes→nsubj→(something), father→compound→Chandrasiri

        The tree shape varies, so we use a general strategy:
        1. Start from head, collect relation words (NOUN tokens whose lemma is a valid predicate)
        2. Follow compound/nsubj/nmod deps to find more relation words
        3. Find the PROPN (proper noun) that's the terminal name

        Returns (chain_of_relation_tokens, name_token_or_None).
        """
        chain = []
        name_token = None

        # Strategy: starting from head, walk through the subtree
        # collecting relation nouns and finding the proper noun name.
        # We follow: compound, nsubj → head, appos, flat deps.
        visited = set()
        queue = [head]

        while queue:
            current = queue.pop(0)
            if current.i in visited or current == poss_token:
                continue
            visited.add(current.i)

            # Classify this token
            lemma = current.lemma_.lower()
            norm = _normalize_predicate(lemma)
            is_relation_word = (norm in VALID_PREDICATES or lemma in RELATION_ALIASES)
            is_propn = current.pos_ == 'PROPN'

            if is_propn and not is_relation_word:
                # This is a name — pick the first one found
                if not name_token:
                    name_token = current
            elif is_relation_word and current.pos_ in ('NOUN', 'PROPN'):
                chain.append(current)

            # Follow structural deps to find connected tokens
            for child in current.children:
                if child == poss_token:
                    continue
                if child.dep_ in ('appos', 'flat', 'flat:name'):
                    # Appositive — this is likely the name
                    if child.pos_ == 'PROPN' and not name_token:
                        name_token = child
                    visited.add(child.i)
                elif child.dep_ in ('compound',):
                    queue.append(child)

            # Check copular patterns involving "be":
            #
            # Pattern A: "my father IS Upananda"
            #   father(nsubj) → is(ROOT) ← Upananda(attr)
            #   → current is nsubj, name is attr of verb
            #
            # Pattern B: "Prabhashi IS my wife"
            #   wife(attr) → is(ROOT) ← Prabhashi(nsubj)
            #   → current is attr, name is nsubj of verb
            if current.dep_ in ('nsubj', 'attr') and current.head.pos_ in ('AUX', 'VERB'):
                cop_verb = current.head
                if cop_verb.lemma_ in ('be',):
                    # Look for the "other side" of the copular
                    target_dep = 'attr' if current.dep_ == 'nsubj' else 'nsubj'
                    for child in cop_verb.children:
                        if child.dep_ == target_dep and child.pos_ == 'PROPN':
                            if not name_token:
                                name_token = child

            # Check naming pattern: "Dog's name is Dexter"
            # current(poss) → name(nsubj) → is(ROOT) ← Dexter(attr)
            # When a possessive points to "name" and "name" is copular,
            # the attr is the entity's actual name.
            if current.dep_ == 'poss' and current.head.lemma_.lower() == 'name':
                name_tok = current.head
                if name_tok.dep_ == 'nsubj' and name_tok.head.lemma_ in ('be',):
                    cop = name_tok.head
                    for child in cop.children:
                        if child.dep_ == 'attr' and child.pos_ == 'PROPN':
                            if not name_token:
                                name_token = child

            # If current is connected upward via compound/nsubj and its
            # head hasn't been visited, follow it
            if current.head != current and current.head.i not in visited:
                if current.dep_ in ('compound', 'nsubj') and current.head != poss_token:
                    # Only follow if head looks like part of the same NP
                    head_lemma = current.head.lemma_.lower()
                    head_norm = _normalize_predicate(head_lemma)
                    if (head_norm in VALID_PREDICATES or
                            current.head.pos_ == 'PROPN' or
                            current.dep_ == 'compound'):
                        queue.append(current.head)

        # Sort chain by position in text
        chain.sort(key=lambda t: t.i)

        # Proximity fallback: if no name was found via the dep tree,
        # check the token immediately after the last chain token.
        # Handles cases where spaCy misclassifies a name as ADV/NOUN
        # (e.g., "my dad upananda" where "upananda" is parsed as advmod).
        if not name_token and chain:
            last_chain = chain[-1]
            next_i = last_chain.i + 1
            if next_i < len(doc):
                candidate = doc[next_i]
                # Accept if: not punctuation, not a common English word,
                # not already identified as a relation word, and passes
                # entity name validation or is a capitalized unknown word
                cand_text = candidate.text.strip()
                cand_lower = candidate.lemma_.lower()
                cand_norm = _normalize_predicate(cand_lower)
                is_relation = cand_norm in VALID_PREDICATES or cand_lower in RELATION_ALIASES
                if (not is_relation
                        and candidate.pos_ not in ('PUNCT', 'SPACE', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'AUX')
                        and cand_lower not in _INVALID_ENTITY_BASE
                        and len(cand_text) >= 3):
                    name_token = candidate

        return chain, name_token

    def _dep_verb_relations(self, doc, memory_id: str, seen: set) -> tuple[list[tuple], list[str]]:
        """
        Find subject-verb-object relations in the dependency tree.

        Patterns:
        - "Dog has/have megaesophagus" → (Dog, has_condition, megaesophagus)
        - "I work at Google" → (user, works_at, Google)
        - "John lives in London" → (John, lives_in, London)

        Returns (results_list, uncertain_snippets).
        """
        results = []
        uncertain = []

        for token in doc:
            if token.pos_ != 'VERB':
                continue

            # Collect subject and objects
            subj_token = None
            obj_token = None
            prep_objs = []  # (prep_word, obj_token)

            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass') and not subj_token:
                    subj_token = child
                elif child.dep_ in ('dobj', 'attr') and not obj_token:
                    obj_token = child
                elif child.dep_ == 'prep':
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            prep_objs.append((child.text.lower(), grandchild))

            if not subj_token:
                continue

            # Resolve subject name
            subj_name = self._resolve_dep_subject(subj_token, doc)
            if not subj_name:
                continue

            # Pattern: "[entity] has/have [condition/thing]"
            if token.lemma_ in ('have',) and obj_token:
                obj_name = self._collect_name_span(obj_token, doc)
                if obj_name and len(obj_name) >= 3:
                    # Skip common non-entity objects
                    if obj_name.lower() not in _INVALID_ENTITY_BASE:
                        results.append((
                            subj_name, "has_condition",
                            obj_name.lower(), "concept", 0.85
                        ))

            # Pattern: "[entity] works/worked at/for [company]"
            if token.lemma_ in ('work',):
                for prep_word, pobj in prep_objs:
                    if prep_word in ('at', 'for'):
                        obj_name = self._collect_name_span(pobj, doc)
                        if obj_name and _is_valid_entity_name(obj_name):
                            results.append((
                                subj_name, "works_at",
                                _cap(obj_name), "company", 0.9
                            ))

            # Pattern: "[entity] lives/stays in [place]"
            if token.lemma_ in ('live', 'stay', 'reside'):
                for prep_word, pobj in prep_objs:
                    if prep_word in ('in', 'at'):
                        obj_name = self._collect_name_span(pobj, doc)
                        if obj_name and _is_valid_entity_name(obj_name):
                            results.append((
                                subj_name, "lives_in",
                                _cap(obj_name), "place", 0.9
                            ))

        return results, uncertain

    def _resolve_dep_subject(self, subj_token, doc) -> Optional[str]:
        """
        Resolve a subject token to an entity name.

        - If subject is "I" or has "my" possessive → user
        - If subject is a PROPN → its name
        - If subject has "my/our" possessive and is a pet word → the pet entity name
        """
        # First-person subject
        if subj_token.text.lower() in ('i', 'we'):
            return self.user_name

        # Check for possessive "my/our"
        has_my = any(
            c.dep_ == 'poss' and c.text.lower() in ('my', 'our')
            for c in subj_token.children
        )

        if has_my:
            # "my Dog" → resolve to the pet entity
            pet_words = {'dog', 'cat', 'pet'}
            if subj_token.lemma_.lower() in pet_words:
                # Look up the user's pet entity name in the graph
                pet_name = subj_token.text.capitalize()
                existing = self.graph.find_entity_by_name(pet_name)
                if existing:
                    return existing.name
                # Check if pet relation exists (may have been created in pass 1)
                user_ent = self.graph.find_entity_by_name(self.user_name)
                if user_ent:
                    row = self.graph.db.execute(
                        "SELECT e.name FROM relations r "
                        "JOIN entities e ON e.id = r.object_id "
                        "WHERE r.subject_id=? AND r.predicate='pet'",
                        (user_ent.id,),
                    ).fetchone()
                    if row:
                        return row[0]
                return pet_name  # Fallback to generic name
            return self.user_name

        # Proper noun subject
        if subj_token.pos_ == 'PROPN':
            return self._collect_name_span(subj_token, doc)

        return None

    def _collect_name_span(self, token, doc) -> str:
        """
        Collect a potentially multi-word name from a token.
        Follows compound and flat deps to build the full name.
        Excludes tokens that are relationship/predicate words (e.g., "father"
        in "father Chandrasiri" where father is compound of Chandrasiri).
        """
        parts = [token]
        for child in token.children:
            if child.dep_ in ('compound', 'flat', 'flat:name'):
                # Exclude relation words from the name span
                child_norm = _normalize_predicate(child.lemma_.lower())
                if child_norm in VALID_PREDICATES or child.lemma_.lower() in RELATION_ALIASES:
                    continue
                parts.append(child)
        parts.sort(key=lambda t: t.i)
        return " ".join(t.text for t in parts)

    @staticmethod
    def _type_from_predicate(predicate: str) -> str:
        """Infer entity type from the relationship predicate."""
        if predicate in ("works_at",):
            return "company"
        elif predicate in ("lives_in", "based_in", "from"):
            return "place"
        elif predicate in ("pet", "dog", "cat"):
            return "animal"
        elif predicate in ("interested_in", "hobby", "studies"):
            return "concept"
        elif predicate in ("has_condition", "diagnosed_with", "is_a"):
            return "concept"
        return "person"

    def _resolve_pronoun(self, pron_token, doc) -> Optional[str]:
        """
        Resolve a third-person pronoun (her/his/their) to the nearest
        preceding entity in the same sentence that could be its antecedent.

        For "My wife was ... with her father Chandrasiri":
          "her" → looks back → finds "wife" (a relation word with poss "my")
          → returns "wife" so the caller can chain: wife.father = father_in_law

        Returns the relation word (lemma) of the antecedent, or None.
        """
        # Gender agreement: her → female relations, his → male relations
        female_rels = {'wife', 'mother', 'sister', 'daughter', 'grandmother', 'aunt'}
        male_rels = {'husband', 'father', 'brother', 'son', 'grandfather', 'uncle'}

        pron = pron_token.text.lower()
        if pron in ('her', 'she'):
            candidate_rels = female_rels
        elif pron in ('his', 'he'):
            candidate_rels = male_rels
        else:
            candidate_rels = female_rels | male_rels  # "their"

        # Scan backwards from the pronoun looking for a relation word
        # that has "my/our" as a possessive modifier
        for i in range(pron_token.i - 1, -1, -1):
            tok = doc[i]
            lemma = tok.lemma_.lower()
            norm = _normalize_predicate(lemma)
            if norm in candidate_rels:
                # Check that this token has "my/our" as possessive
                has_my = any(
                    c.dep_ == 'poss' and c.text.lower() in ('my', 'our')
                    for c in tok.children
                )
                if has_my:
                    return lemma
        return None

    def _check_pet_compound(self, head_token) -> bool:
        """
        Check if head_token has a compound child that's a pet word.
        Handles: "My Dog dexter" → head=dexter, Dog is compound child.
        """
        pet_words = {'dog', 'cat', 'pet'}
        for child in head_token.children:
            if child.dep_ == 'compound' and child.lemma_.lower() in pet_words:
                return True
        return False

    # ── Tier 2: LLM resolution for uncertain extractions ──────

    def _resolve_uncertain_with_llm(self, uncertain: list[str], result: dict, memory_id: str):
        """
        Tier 2: Pass uncertain text snippets to the LLM for extraction.

        Only called when Tier 1 (dep parsing) couldn't fully resolve
        a structural pattern — e.g., a possessive relation where no
        proper name was found, or a complex multi-clause sentence.
        """
        if not uncertain or not self.llm_fn:
            return

        combined = "\n".join(uncertain)
        llm_result = self.extract_with_llm(combined, memory_id)
        result["entities"].extend(llm_result.get("entities", []))
        result["relations"].extend(llm_result.get("relations", []))

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
                        elif predicate in ("has_condition", "diagnosed_with", "is_a"):
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

    def _resolve_generic_entity(self, generic_name: str) -> Optional[str]:
        """Resolve generic LLM labels like 'Wife' to actual names from the graph.

        Uses VALID_PREDICATES and RELATION_ALIASES — no hardcoded mapping.
        Adding a new predicate automatically makes it resolvable.
        """
        predicate = _normalize_predicate(generic_name)
        if predicate not in VALID_PREDICATES:
            return None
        user_entity = self.graph.find_entity_by_name(self.user_name)
        if not user_entity:
            return None
        row = self.graph.db.execute(
            "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
            (user_entity.id, predicate),
        ).fetchone()
        if row:
            resolved = self.graph.db.execute(
                "SELECT name FROM entities WHERE id=?", (row[0],),
            ).fetchone()
            if resolved:
                return resolved[0]
        return None

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

    def _get_graph_context_for_extraction(self, text: str) -> str:
        """
        Build a compact summary of the current graph neighborhood relevant
        to the text being extracted. This lets the LLM:
        1. Resolve against EXISTING entities (avoid duplicates)
        2. Understand existing relationships (avoid contradictions)
        3. Merge entities it recognizes as the same (Dog → Dexter)

        Returns a short string suitable for injection into the extraction prompt.
        """
        lines = []

        # Get all entities (limit to keep prompt short for 8B models)
        try:
            rows = self.graph.db.execute(
                "SELECT id, name, entity_type FROM entities ORDER BY name LIMIT 30"
            ).fetchall()
        except Exception:
            return ""

        if not rows:
            return ""

        entity_map = {row[0]: (row[1], row[2]) for row in rows}
        entity_names = {row[1].lower() for row in rows}

        lines.append("EXISTING ENTITIES:")
        for eid, (name, etype) in entity_map.items():
            lines.append(f"  {name} ({etype})")

        # Get relations involving these entities
        try:
            rel_rows = self.graph.db.execute(
                "SELECT subject_id, predicate, object_id FROM relations "
                "WHERE is_inferred = 0 LIMIT 40"
            ).fetchall()
        except Exception:
            rel_rows = []

        if rel_rows:
            lines.append("EXISTING RELATIONS:")
            for subj_id, pred, obj_id in rel_rows:
                subj = entity_map.get(subj_id, (str(subj_id), "?"))
                obj = entity_map.get(obj_id, (str(obj_id), "?"))
                lines.append(f"  {subj[0]} --{pred}--> {obj[0]}")

        return "\n".join(lines)

    def extract_with_llm(self, text: str, memory_id: str = "") -> dict:
        """
        Graph-aware LLM extraction. Sends the current graph neighborhood
        alongside the text so the LLM can:
        - Resolve against existing entities (no duplicates)
        - Spot redundant entities (Dog = Dexter)
        - Build proper relationships using known context
        """
        if not self.llm_fn:
            return {"entities": [], "relations": []}

        graph_ctx = self._get_graph_context_for_extraction(text)

        # Keep prompt MINIMAL — small quantized models (7-8B) choke on
        # long structured instructions. Only include graph context if small.
        entity_names = ""
        if graph_ctx:
            # Only inject entity names (not full relations) to keep it short
            names_only = [line.strip() for line in graph_ctx.split("\n")
                          if line.strip() and not line.startswith("EXISTING")]
            if len(names_only) <= 15:
                entity_names = f"Known entities: {', '.join(n.split('(')[0].strip() for n in names_only[:15])}\n"

        prompt = (
            f"Extract people, animals, places from this text. User = {self.user_name}.\n"
            f"{entity_names}"
            f'Text: "{text[:500]}"\n\n'
            f"Output ONLY these lines (nothing else):\n"
            f"ENTITY: Name | type\n"
            f"RELATION: Subject | predicate | Object\n"
            f"Types: person, animal, place, condition\n"
            f"Predicates: father, mother, wife, husband, pet, has_condition, lives_in, works_at"
        )

        try:
            logger.info(f"Calling LLM for entity extraction ({len(prompt)} chars prompt)")
            result = self.llm_fn(prompt)
            logger.info(f"LLM entity extraction response ({len(result)} chars): {result[:200]}")

            # If empty response, try minimal prompt (just ask for entities)
            if not result.strip():
                logger.info("Empty response — retrying with minimal prompt")
                result = self.llm_fn(
                    f"List people and places in: \"{text[:300]}\"\n"
                    f"Format: ENTITY: Name | person\nRELATION: {self.user_name} | predicate | Name"
                )
                logger.info(f"Retry response ({len(result)} chars): {result[:200]}")
        except Exception as e:
            logger.error(f"LLM entity extraction FAILED: {e}", exc_info=True)
            return {"entities": [], "relations": []}

        if not result.strip() or "NONE" in result.upper()[:20]:
            logger.info("LLM returned no entities")
            return {"entities": [], "relations": []}

        entities = []
        relations = []
        merges = []

        for line in result.strip().split("\n"):
            line = line.strip()

            if line.startswith("ENTITY:"):
                parts = line[7:].strip().split("|")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    etype = parts[1].strip().lower()
                    if not _is_valid_entity_name(name):
                        continue
                    if etype not in ("person", "company", "place", "animal", "concept", "condition"):
                        etype = "unknown"
                    entity = self.graph.add_entity(Entity(
                        name=name, entity_type=etype,
                        source_memory_id=memory_id,
                    ))
                    entities.append(entity)

            elif line.startswith("MERGE:"):
                # LLM detected two entities are the same (e.g., "Dog" and "Dexter")
                parts = line[6:].strip().split("|")
                if len(parts) >= 2:
                    old_name = parts[0].strip()
                    new_name = parts[1].strip()
                    if old_name and new_name and old_name.lower() != new_name.lower():
                        old_ent = self.graph.find_entity_by_name(old_name)
                        new_ent = self.graph.find_entity_by_name(new_name)
                        if old_ent and new_ent:
                            try:
                                self.graph.merge_entities(old_ent.id, new_ent.id)
                                merges.append((old_name, new_name))
                                logger.info(f"LLM-directed merge: {old_name} → {new_name}")
                            except Exception as e:
                                logger.warning(f"LLM merge failed ({old_name} → {new_name}): {e}")
                        elif old_ent and not new_ent:
                            # New name doesn't exist yet — rename the old entity
                            try:
                                self.graph.db.execute(
                                    "UPDATE entities SET name=? WHERE id=?",
                                    (new_name, old_ent.id),
                                )
                                self.graph.db.commit()
                                merges.append((old_name, new_name))
                                logger.info(f"LLM-directed rename: {old_name} → {new_name}")
                            except Exception as e:
                                logger.warning(f"LLM rename failed ({old_name} → {new_name}): {e}")

            elif line.startswith("RELATION:"):
                parts = line[9:].strip().split("|")
                if len(parts) >= 3:
                    subj_name = parts[0].strip()
                    predicate = _normalize_predicate(parts[1].strip())
                    obj_name = parts[2].strip()
                    if predicate not in VALID_PREDICATES:
                        logger.debug(f"Rejected LLM predicate: {predicate}")
                        continue
                    if not subj_name or not obj_name or len(obj_name) < 2:
                        continue

                    # Resolve user references
                    if subj_name.lower() in ("user", self.user_name.lower()):
                        subj_name = self.user_name

                    # If either name is a known predicate/relationship word
                    # (e.g., "Wife", "Dog", "Boss"), try to resolve it to a
                    # real entity name. Derived from VALID_PREDICATES so adding
                    # new predicates auto-protects against this.
                    for ref_name, setter in [("subj", subj_name), ("obj", obj_name)]:
                        normalized = _normalize_predicate(setter)
                        if normalized in VALID_PREDICATES:
                            resolved = self._resolve_generic_entity(setter)
                            if resolved:
                                if ref_name == "subj":
                                    subj_name = resolved
                                else:
                                    obj_name = resolved

                    if not _is_valid_entity_name(subj_name):
                        logger.debug(f"Rejected invalid subj_name: {subj_name}")
                        continue
                    if not _is_valid_entity_name(obj_name):
                        logger.debug(f"Rejected invalid obj_name: {obj_name}")
                        continue

                    subj = self.graph.find_entity_by_name(subj_name)
                    if not subj:
                        subj = self.graph.add_entity(Entity(name=subj_name, entity_type="person"))
                        if not subj:
                            continue
                    obj = self.graph.find_entity_by_name(obj_name)
                    if not obj:
                        obj = self.graph.add_entity(Entity(name=obj_name))

                    rel = self.graph.add_relation(Relation(
                        subject_id=subj.id, predicate=predicate,
                        object_id=obj.id, source_memory_id=memory_id,
                    ))
                    relations.append(rel)

        return {"entities": entities, "relations": relations, "merges": merges}
