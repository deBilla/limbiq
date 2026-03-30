"""Per-entity persistent state — distributed cellular memory.

Each entity in the knowledge graph maintains its own state:
- Resting activation (like resting membrane potential)
- Signal history (like epigenetic marks)
- Receptor density (like cell surface receptor count)
- Expression mask (like DNA methylation — same data, different expression)
- Sentinel pattern (like immune T-cell memory)

Thread-safe: uses MemoryStore's per-thread db property.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default receptor densities — neutral sensitivity to all signals.
_DEFAULT_RECEPTOR_DENSITY = {
    "dopamine": 1.0,
    "gaba": 1.0,
    "serotonin": 1.0,
    "acetylcholine": 1.0,
    "norepinephrine": 1.0,
}

# Receptor density bounds — prevents runaway positive feedback.
RECEPTOR_MIN = 0.1
RECEPTOR_MAX = 3.0

# How much receptor density changes per signal hit.
RECEPTOR_ADAPTATION_RATE = 0.05


@dataclass
class EntityState:
    """Persistent per-entity state — the cell's memory."""

    entity_id: str = ""
    resting_activation: float = 0.0
    signal_history: dict[str, int] = field(default_factory=dict)
    receptor_density: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RECEPTOR_DENSITY)
    )
    expression_mask: dict[str, bool] = field(default_factory=dict)
    total_activations: int = 0
    last_activated_at: float = 0.0
    sentinel_pattern: str | None = None
    last_consolidated_at: float = 0.0


class EntityStateStore:
    """SQLite-backed per-entity state store.

    Follows the same pattern as ClusterStore and RuleStore —
    wraps a SQLite table, uses MemoryStore for thread-safe DB access.
    """

    def __init__(self, memory_store):
        self._store = memory_store
        self._init_tables()

    @property
    def db(self):
        return self._store.db

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS entity_state (
                entity_id TEXT PRIMARY KEY,
                resting_activation REAL DEFAULT 0.0,
                signal_history TEXT DEFAULT '{}',
                receptor_density TEXT DEFAULT '{}',
                expression_mask TEXT DEFAULT '{}',
                total_activations INTEGER DEFAULT 0,
                last_activated_at REAL DEFAULT 0,
                sentinel_pattern TEXT DEFAULT NULL,
                last_consolidated_at REAL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entity_state_activation
                ON entity_state(resting_activation DESC);
            CREATE INDEX IF NOT EXISTS idx_entity_state_last_consolidated
                ON entity_state(last_consolidated_at);
        """)
        self.db.commit()

    # ── Read ──────────────────────────────────────────────────

    def get_state(self, entity_id: str) -> EntityState:
        """Get state for an entity. Returns default state if none exists."""
        row = self.db.execute(
            "SELECT entity_id, resting_activation, signal_history, "
            "receptor_density, expression_mask, total_activations, "
            "last_activated_at, sentinel_pattern, last_consolidated_at "
            "FROM entity_state WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if row:
            return self._row_to_state(row)
        return EntityState(entity_id=entity_id)

    def get_all_states(self) -> list[EntityState]:
        """Get all entity states, ordered by resting activation descending."""
        rows = self.db.execute(
            "SELECT entity_id, resting_activation, signal_history, "
            "receptor_density, expression_mask, total_activations, "
            "last_activated_at, sentinel_pattern, last_consolidated_at "
            "FROM entity_state ORDER BY resting_activation DESC"
        ).fetchall()
        return [self._row_to_state(r) for r in rows]

    def get_top_activated(self, limit: int = 20) -> list[EntityState]:
        """Get entities with highest resting activation."""
        rows = self.db.execute(
            "SELECT entity_id, resting_activation, signal_history, "
            "receptor_density, expression_mask, total_activations, "
            "last_activated_at, sentinel_pattern, last_consolidated_at "
            "FROM entity_state ORDER BY resting_activation DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_state(r) for r in rows]

    def get_sentinels(self) -> list[EntityState]:
        """Get all entities with active sentinel patterns."""
        rows = self.db.execute(
            "SELECT entity_id, resting_activation, signal_history, "
            "receptor_density, expression_mask, total_activations, "
            "last_activated_at, sentinel_pattern, last_consolidated_at "
            "FROM entity_state WHERE sentinel_pattern IS NOT NULL"
        ).fetchall()
        return [self._row_to_state(r) for r in rows]

    # ── Write ─────────────────────────────────────────────────

    def ensure_state_exists(self, entity_id: str) -> None:
        """Create default state for an entity if it doesn't exist yet."""
        existing = self.db.execute(
            "SELECT 1 FROM entity_state WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if not existing:
            self.db.execute(
                "INSERT INTO entity_state "
                "(entity_id, resting_activation, signal_history, receptor_density, "
                "expression_mask, total_activations, last_activated_at, "
                "sentinel_pattern, last_consolidated_at, updated_at) "
                "VALUES (?, 0.0, '{}', ?, '{}', 0, 0, NULL, 0, ?)",
                (entity_id, json.dumps(_DEFAULT_RECEPTOR_DENSITY), time.time()),
            )
            self.db.commit()

    def activate(self, entity_id: str, delta: float = 0.1) -> None:
        """Increase resting activation and bump activation counter.

        Like depolarizing a cell — makes it more likely to fire next time.
        Activation is clamped to [0, 2.0].
        """
        self.ensure_state_exists(entity_id)
        now = time.time()
        self.db.execute(
            "UPDATE entity_state SET "
            "resting_activation = MIN(2.0, MAX(0, resting_activation + ?)), "
            "total_activations = total_activations + 1, "
            "last_activated_at = ?, "
            "updated_at = ? "
            "WHERE entity_id = ?",
            (delta, now, now, entity_id),
        )
        self.db.commit()

    def record_signal(self, entity_id: str, signal_type: str) -> None:
        """Record that a signal affected this entity and adapt receptor density.

        Repeated exposure to a signal upregulates the receptor —
        like biological receptor upregulation from chronic stimulation.
        """
        self.ensure_state_exists(entity_id)
        state = self.get_state(entity_id)

        # Increment signal count
        history = state.signal_history
        history[signal_type] = history.get(signal_type, 0) + 1

        # Adapt receptor density (upregulate on repeated exposure)
        density = state.receptor_density
        current = density.get(signal_type, 1.0)
        adapted = min(RECEPTOR_MAX, current + RECEPTOR_ADAPTATION_RATE)
        density[signal_type] = round(adapted, 4)

        self.db.execute(
            "UPDATE entity_state SET "
            "signal_history = ?, receptor_density = ?, updated_at = ? "
            "WHERE entity_id = ?",
            (json.dumps(history), json.dumps(density), time.time(), entity_id),
        )
        self.db.commit()

    def set_sentinel(self, entity_id: str, pattern: str | None) -> None:
        """Set or clear a sentinel pattern for an entity.

        Sentinels are immune-memory-style watchers that trigger on
        specific patterns, like T-cells recognizing an antigen.
        """
        self.ensure_state_exists(entity_id)
        self.db.execute(
            "UPDATE entity_state SET sentinel_pattern = ?, updated_at = ? "
            "WHERE entity_id = ?",
            (pattern, time.time(), entity_id),
        )
        self.db.commit()

    def update_expression_mask(self, entity_id: str, mask: dict[str, bool]) -> None:
        """Update which properties are 'expressed' for this entity.

        Like DNA methylation — doesn't change the data, just which
        aspects are active during retrieval.
        """
        self.ensure_state_exists(entity_id)
        self.db.execute(
            "UPDATE entity_state SET expression_mask = ?, updated_at = ? "
            "WHERE entity_id = ?",
            (json.dumps(mask), time.time(), entity_id),
        )
        self.db.commit()

    # ── Batch operations ──────────────────────────────────────

    def decay_activations(self, decay_factor: float = 0.95) -> int:
        """Decay all resting activations by a factor.

        Called at end_session() — like ion channels resetting between
        periods of activity. Returns count of entities decayed.
        """
        result = self.db.execute(
            "UPDATE entity_state SET "
            "resting_activation = resting_activation * ?, "
            "updated_at = ? "
            "WHERE resting_activation > 0.001",
            (decay_factor, time.time()),
        )
        self.db.commit()
        # Zero out very small activations to avoid infinite decay
        self.db.execute(
            "UPDATE entity_state SET resting_activation = 0 "
            "WHERE resting_activation <= 0.001 AND resting_activation > 0"
        )
        self.db.commit()
        return result.rowcount

    def decay_receptor_density(self, decay_rate: float = 0.01) -> None:
        """Decay all receptor densities toward baseline (1.0).

        Called during deep rest — prevents runaway receptor upregulation.
        """
        rows = self.db.execute(
            "SELECT entity_id, receptor_density FROM entity_state"
        ).fetchall()
        for entity_id, density_json in rows:
            density = json.loads(density_json) if density_json else {}
            changed = False
            for signal, value in density.items():
                if value > 1.0:
                    density[signal] = round(max(1.0, value - decay_rate), 4)
                    changed = True
                elif value < 1.0:
                    density[signal] = round(min(1.0, value + decay_rate), 4)
                    changed = True
            if changed:
                self.db.execute(
                    "UPDATE entity_state SET receptor_density = ?, updated_at = ? "
                    "WHERE entity_id = ?",
                    (json.dumps(density), time.time(), entity_id),
                )
        self.db.commit()

    def cleanup_orphaned(self) -> int:
        """Remove state for entities that no longer exist in the graph.

        Like cellular apoptosis — cleanup after entity deletion.
        """
        result = self.db.execute(
            "DELETE FROM entity_state WHERE entity_id NOT IN "
            "(SELECT id FROM entities)"
        )
        self.db.commit()
        return result.rowcount

    # ── Internal ──────────────────────────────────────────────

    def _row_to_state(self, row) -> EntityState:
        return EntityState(
            entity_id=row[0],
            resting_activation=row[1] or 0.0,
            signal_history=json.loads(row[2]) if row[2] else {},
            receptor_density=json.loads(row[3]) if row[3] else dict(_DEFAULT_RECEPTOR_DENSITY),
            expression_mask=json.loads(row[4]) if row[4] else {},
            total_activations=row[5] or 0,
            last_activated_at=row[6] or 0.0,
            sentinel_pattern=row[7],
            last_consolidated_at=row[8] or 0.0,
        )
