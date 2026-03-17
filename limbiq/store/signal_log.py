"""Signal event history storage.

Thread-safe: gets the db connection from the store on each call
so it uses the per-thread connection via threading.local().
"""

import json
import uuid

from limbiq.types import SignalEvent


class SignalLog:
    def __init__(self, store):
        """Accept the MemoryStore (not a raw connection) for thread-safe access."""
        self._store = store

    def log(self, event: SignalEvent) -> None:
        db = self._store.db
        db.execute(
            """INSERT INTO signal_log (id, signal_type, trigger, timestamp, details, memory_ids_affected)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                event.signal_type.value if hasattr(event.signal_type, "value") else event.signal_type,
                event.trigger,
                event.timestamp,
                json.dumps(event.details),
                json.dumps(event.memory_ids_affected),
            ),
        )
        db.commit()

    def get_recent(self, limit: int = 50) -> list[SignalEvent]:
        db = self._store.db
        cursor = db.execute(
            "SELECT signal_type, trigger, timestamp, details, memory_ids_affected "
            "FROM signal_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        events = []
        for row in cursor.fetchall():
            events.append(
                SignalEvent(
                    signal_type=row[0],
                    trigger=row[1],
                    timestamp=row[2],
                    details=json.loads(row[3]),
                    memory_ids_affected=json.loads(row[4]),
                )
            )
        return events
