"""
OpenTelemetry instrumentation for Limbiq.

Provides:
- Traces: spans for process(), observe(), query_graph(), reason(), propagate()
- Metrics: counters, histograms, gauges for memory/graph/signal operations
- In-memory span collector for the dashboard (no external Jaeger required)
"""

import time
import functools
import logging
from collections import deque
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


# ─── In-Memory Span Collector ──────────────────────────────────
# Stores recent spans so the dashboard can display them without Jaeger.

@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    status: str
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


class InMemorySpanExporter:
    """Collects spans into a deque for dashboard consumption."""

    def __init__(self, maxlen: int = 1000):
        self.spans: deque[SpanRecord] = deque(maxlen=maxlen)
        self._shutdown = False

    def export(self, spans):
        if self._shutdown:
            return
        for span in spans:
            ctx = span.get_span_context()
            parent = span.parent
            record = SpanRecord(
                trace_id=format(ctx.trace_id, "032x"),
                span_id=format(ctx.span_id, "016x"),
                parent_span_id=format(parent.span_id, "016x") if parent else None,
                name=span.name,
                start_time_ns=span.start_time,
                end_time_ns=span.end_time,
                duration_ms=(span.end_time - span.start_time) / 1e6,
                status=span.status.status_code.name if span.status else "UNSET",
                attributes=dict(span.attributes) if span.attributes else {},
                events=[
                    {"name": e.name, "timestamp_ns": e.timestamp,
                     "attributes": dict(e.attributes) if e.attributes else {}}
                    for e in (span.events or [])
                ],
            )
            self.spans.append(record)

    def shutdown(self):
        self._shutdown = True

    def force_flush(self, timeout_millis=None):
        pass

    def get_recent(self, limit: int = 100) -> list[dict]:
        items = list(self.spans)[-limit:]
        return [asdict(s) for s in items]

    def get_traces(self, limit: int = 50) -> list[dict]:
        """Group spans by trace_id."""
        traces = {}
        for s in self.spans:
            traces.setdefault(s.trace_id, []).append(asdict(s))
        # Return most recent traces
        result = []
        for trace_id, spans in list(traces.items())[-limit:]:
            root = min(spans, key=lambda s: s["start_time_ns"])
            result.append({
                "trace_id": trace_id,
                "root_span": root["name"],
                "span_count": len(spans),
                "duration_ms": max(s["end_time_ns"] for s in spans) - min(s["start_time_ns"] for s in spans),
                "spans": sorted(spans, key=lambda s: s["start_time_ns"]),
            })
        return result


# ─── Metrics Collector ────────────────────────────────────────

class LimbiqMetrics:
    """Central metrics registry for Limbiq operations."""

    def __init__(self, meter: metrics.Meter):
        # Counters
        self.process_count = meter.create_counter(
            "limbiq.process.total",
            description="Total process() calls",
        )
        self.observe_count = meter.create_counter(
            "limbiq.observe.total",
            description="Total observe() calls",
        )
        self.query_count = meter.create_counter(
            "limbiq.query.total",
            description="Total graph queries",
        )
        self.signal_count = meter.create_counter(
            "limbiq.signals.fired.total",
            description="Total signals fired",
        )
        self.entity_extracted_count = meter.create_counter(
            "limbiq.entities.extracted.total",
            description="Entities extracted from text",
        )
        self.memory_ops_count = meter.create_counter(
            "limbiq.memory.ops.total",
            description="Memory operations (store, suppress, restore)",
        )

        # Histograms
        self.process_duration = meter.create_histogram(
            "limbiq.process.duration_ms",
            description="Process latency in milliseconds",
            unit="ms",
        )
        self.query_duration = meter.create_histogram(
            "limbiq.query.duration_ms",
            description="Graph query latency in milliseconds",
            unit="ms",
        )
        self.query_confidence = meter.create_histogram(
            "limbiq.query.confidence",
            description="Query answer confidence scores",
        )

        # Time series storage for dashboard charts
        self._time_series: deque = deque(maxlen=3600)  # 1 hour at 1/sec

    def record_process(self, duration_ms: float, memories_retrieved: int):
        self.process_count.add(1)
        self.process_duration.record(duration_ms)
        self._time_series.append({
            "ts": time.time(),
            "type": "process",
            "duration_ms": duration_ms,
            "memories": memories_retrieved,
        })

    def record_query(self, duration_ms: float, answered: bool, confidence: float):
        self.query_count.add(1, {"answered": str(answered)})
        self.query_duration.record(duration_ms)
        if confidence > 0:
            self.query_confidence.record(confidence)
        self._time_series.append({
            "ts": time.time(),
            "type": "query",
            "duration_ms": duration_ms,
            "answered": answered,
            "confidence": confidence,
        })

    def record_signal(self, signal_type: str, trigger: str = ""):
        self.signal_count.add(1, {"signal_type": signal_type})
        self._time_series.append({
            "ts": time.time(),
            "type": "signal",
            "signal_type": signal_type,
            "trigger": trigger,
        })

    def record_entity_extraction(self, count: int):
        self.entity_extracted_count.add(count)

    def record_memory_op(self, operation: str):
        self.memory_ops_count.add(1, {"operation": operation})

    def get_time_series(self, since: float = 0, metric_type: str = None) -> list[dict]:
        result = []
        for point in self._time_series:
            if point["ts"] > since:
                if metric_type is None or point["type"] == metric_type:
                    result.append(point)
        return result


# ─── Setup ────────────────────────────────────────────────────

_span_exporter: Optional[InMemorySpanExporter] = None
_metrics: Optional[LimbiqMetrics] = None
_tracer: Optional[trace.Tracer] = None


def setup_telemetry(service_name: str = "limbiq") -> None:
    """Initialize OpenTelemetry with in-memory exporters for the dashboard."""
    global _span_exporter, _metrics, _tracer

    resource = Resource.create({"service.name": service_name})

    # Traces — in-memory collector (no external dependency)
    _span_exporter = InMemorySpanExporter(maxlen=2000)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_span_exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("limbiq")

    # Metrics
    meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter("limbiq")
    _metrics = LimbiqMetrics(meter)

    logger.info("Telemetry initialized (in-memory spans + metrics)")


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("limbiq")
    return _tracer


def get_metrics() -> Optional[LimbiqMetrics]:
    return _metrics


def get_span_exporter() -> Optional[InMemorySpanExporter]:
    return _span_exporter


# ─── Decorator for auto-instrumentation ─────────────────────

def traced(span_name: str = None, record_args: list[str] = None):
    """Decorator to auto-trace a function with OpenTelemetry."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"limbiq.{fn.__qualname__}"
            with tracer.start_as_current_span(name) as span:
                # Record specified arguments as attributes
                if record_args:
                    for arg_name in record_args:
                        if arg_name in kwargs:
                            val = kwargs[arg_name]
                            if isinstance(val, (str, int, float, bool)):
                                span.set_attribute(f"limbiq.{arg_name}", val)

                t0 = time.time()
                try:
                    result = fn(*args, **kwargs)
                    span.set_attribute("limbiq.duration_ms", (time.time() - t0) * 1000)
                    return result
                except Exception as e:
                    span.set_attribute("limbiq.error", str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
