"""Signal event history storage."""

import json
import sqlite3
import uuid

from limbiq.types import SignalEvent


class SignalLog:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def log(self, event: SignalEvent) -> None:
        self.db.execute(
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
        self.db.commit()

    def get_recent(self, limit: int = 50) -> list[SignalEvent]:
        cursor = self.db.execute(
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
