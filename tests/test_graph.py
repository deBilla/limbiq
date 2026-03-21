"""Tests for the knowledge graph module."""

import tempfile
import shutil

import pytest

from limbiq import Limbiq
from limbiq.graph.store import GraphStore, Entity, Relation
from limbiq.graph.entities import EntityExtractor
from limbiq.graph.inference import InferenceEngine
from limbiq.graph.query import GraphQuery


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def lq(tmp_dir):
    return Limbiq(store_path=tmp_dir, user_id="Dimuthu")


# ── GraphStore ────────────────────────────────────────────────


def test_add_entity(lq):
    entity = Entity(name="Upananda", entity_type="person")
    result = lq._core.graph.add_entity(entity)
    assert result.name == "Upananda"


def test_entity_dedup(lq):
    e1 = lq._core.graph.add_entity(Entity(name="Upananda", entity_type="person"))
    e2 = lq._core.graph.add_entity(Entity(name="upananda", entity_type="person"))
    assert e1.id == e2.id  # Case-insensitive dedup


def test_add_relation(lq):
    graph = lq._core.graph
    subj = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    obj = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    rel = graph.add_relation(Relation(
        subject_id=subj.id, predicate="father", object_id=obj.id,
    ))
    assert rel.predicate == "father"

    relations = graph.get_relations_for(subj.id)
    assert len(relations) == 1


def test_relation_dedup(lq):
    graph = lq._core.graph
    subj = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    obj = graph.add_entity(Entity(name="Prabhashi", entity_type="person"))
    graph.add_relation(Relation(
        subject_id=subj.id, predicate="wife", object_id=obj.id,
    ))
    graph.add_relation(Relation(
        subject_id=subj.id, predicate="wife", object_id=obj.id,
    ))
    relations = graph.get_all_relations()
    wife_rels = [r for r in relations if r.predicate == "wife"]
    assert len(wife_rels) == 1


# ── EntityExtractor ───────────────────────────────────────────


def test_heuristic_extract_my_father(lq):
    extractor = lq._core.entity_extractor
    result = extractor.extract_from_memory("My father is Upananda")
    assert any(e.name == "Upananda" for e in result["entities"])
    assert any(r.predicate == "father" for r in result["relations"])


def test_heuristic_extract_reverse(lq):
    extractor = lq._core.entity_extractor
    result = extractor.extract_from_memory("Prabhashi is my wife")
    assert any(e.name == "Prabhashi" for e in result["entities"])
    assert any(r.predicate == "wife" for r in result["relations"])


def test_heuristic_extract_compressed_format(lq):
    extractor = lq._core.entity_extractor
    result = extractor.extract_from_memory("User's father is Upananda")
    assert any(r.predicate == "father" for r in result["relations"])


def test_heuristic_extract_works_at(lq):
    extractor = lq._core.entity_extractor
    result = extractor.extract_from_memory("I work at Bitsmedia")
    assert any(r.predicate == "works_at" for r in result["relations"])


# ── InferenceEngine ───────────────────────────────────────────


def test_inference_father_in_law(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    father = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    wife = graph.add_entity(Entity(name="Prabhashi", entity_type="person"))

    graph.add_relation(Relation(
        subject_id=user.id, predicate="father", object_id=father.id,
    ))
    graph.add_relation(Relation(
        subject_id=user.id, predicate="wife", object_id=wife.id,
    ))

    engine = lq._core.inference_engine
    inferred = engine.run_full_inference()
    assert inferred > 0

    # Should find father-in-law relationship
    result = engine.query_relationship("Upananda", "Prabhashi")
    assert result["found"]
    assert any("father-in-law" in r["description"].lower() for r in result["relations"])


def test_no_inference_without_matching_rules(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    company = graph.add_entity(Entity(name="Bitsmedia", entity_type="company"))

    graph.add_relation(Relation(
        subject_id=user.id, predicate="works_at", object_id=company.id,
    ))

    engine = lq._core.inference_engine
    inferred = engine.run_full_inference()
    assert inferred == 0


# ── GraphQuery ────────────────────────────────────────────────


def test_query_fact(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    father = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    graph.add_relation(Relation(
        subject_id=user.id, predicate="father", object_id=father.id,
    ))

    result = lq.query_graph("What's my father's name?")
    assert result["answered"]
    assert "Upananda" in result["answer"]


def test_query_entity(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    father = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    graph.add_relation(Relation(
        subject_id=user.id, predicate="father", object_id=father.id,
    ))

    result = lq.query_graph("Who is Upananda?")
    assert result["answered"]
    assert "father" in result["answer"].lower()


# ── World Summary ─────────────────────────────────────────────


def test_world_summary(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    father = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    wife = graph.add_entity(Entity(name="Prabhashi", entity_type="person"))
    company = graph.add_entity(Entity(name="Bitsmedia", entity_type="company"))

    graph.add_relation(Relation(subject_id=user.id, predicate="father", object_id=father.id))
    graph.add_relation(Relation(subject_id=user.id, predicate="wife", object_id=wife.id))
    graph.add_relation(Relation(subject_id=user.id, predicate="works_at", object_id=company.id))

    lq._core.inference_engine.run_full_inference()
    summary = lq.get_world_summary()

    assert "Upananda" in summary
    assert "Prabhashi" in summary
    assert "Bitsmedia" in summary
    # Should be compact
    assert len(summary.split()) < 100


def test_world_summary_includes_inferred(lq):
    graph = lq._core.graph
    user = graph.add_entity(Entity(name="Dimuthu", entity_type="person"))
    father = graph.add_entity(Entity(name="Upananda", entity_type="person"))
    wife = graph.add_entity(Entity(name="Prabhashi", entity_type="person"))

    graph.add_relation(Relation(subject_id=user.id, predicate="father", object_id=father.id))
    graph.add_relation(Relation(subject_id=user.id, predicate="wife", object_id=wife.id))

    lq._core.inference_engine.run_full_inference()
    summary = lq.get_world_summary()

    assert "father-in-law" in summary.lower()


# ── Integration via observe() ─────────────────────────────────


def test_observe_extracts_entities(lq):
    lq.observe("My father is Upananda", "Nice to know!")
    entities = lq.get_entities()
    names = [e.name for e in entities]
    assert "Upananda" in names


def test_end_session_runs_inference(lq):
    lq.start_session()
    lq.observe("My father is Upananda", "...")
    lq.observe("My wife is Prabhashi", "...")
    results = lq.end_session()
    assert results.get("graph_inferred", 0) > 0


# ── Graph Stats ───────────────────────────────────────────────


def test_graph_stats(lq):
    stats = lq.get_graph_stats()
    assert "entities" in stats
    assert "relations" in stats
    assert "inferred" in stats
