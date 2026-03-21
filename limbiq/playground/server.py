"""
FastAPI server for Limbiq Playground.

Creates the app, mounts API routes, and serves the React SPA.
"""

import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from limbiq.playground.api import router as api_router

logger = logging.getLogger(__name__)


def create_app(
    store_path: str = "./data/limbiq",
    user_id: str = "default",
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_client=None,
    search_client=None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup/shutdown lifecycle."""
        from limbiq import Limbiq

        logger.info(f"Initializing Limbiq (store={store_path}, user={user_id})")
        app.state.lq = Limbiq(
            store_path=store_path,
            user_id=user_id,
            embedding_model=embedding_model,
            llm_fn=llm_client,
        )
        app.state.llm = llm_client
        app.state.search_client = search_client
        app.state.start_time = time.time()

        # Auto-instrument FastAPI with OpenTelemetry
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:
            logger.warning(f"FastAPI instrumentation skipped: {e}")

        logger.info("Limbiq Playground ready")
        yield

        logger.info("Shutting down Limbiq Playground")

    app = FastAPI(
        title="Limbiq Playground",
        version="0.1.0",
        description="Interactive dashboard for exploring limbiq's knowledge graph, "
                    "memory system, and neurotransmitter signals.",
        lifespan=lifespan,
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API
    app.include_router(api_router, prefix="/api/v1")

    # Serve React SPA
    @app.get("/", response_class=HTMLResponse)
    async def serve_spa():
        return _get_spa_html()

    # Catch-all for SPA client-side routing
    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa_fallback(path: str):
        # Don't intercept API routes
        if path.startswith("api/"):
            return None
        return _get_spa_html()

    return app


def _get_spa_html() -> str:
    """Return the React SPA HTML.

    Checks for a pre-built bundle first, falls back to the inline SPA.
    """
    # Check for pre-built React bundle
    build_dir = Path(__file__).parent / "build" / "dist"
    index = build_dir / "index.html"
    if index.exists():
        return index.read_text()

    # Inline React SPA (no build step needed)
    return _INLINE_SPA


# ─── Inline React SPA ─────────────────────────────────────────
# Single HTML file with React, D3, and Recharts loaded from CDN.
# This means `python -m limbiq.playground` works with ZERO frontend build.

_INLINE_SPA = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Limbiq Playground</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.9/babel.min.js"></script>
  <style>
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
      --green: #3fb950; --red: #f85149; --yellow: #d29922;
      --purple: #bc8cff; --orange: #f0883e; --cyan: #39d2c0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
           background: var(--bg); color: var(--text); }
    .app { display: grid; grid-template-columns: 220px 1fr; grid-template-rows: 56px 1fr; height: 100vh; }
    .header { grid-column: 1/-1; background: var(--surface); border-bottom: 1px solid var(--border);
              display: flex; align-items: center; padding: 0 20px; gap: 16px; }
    .header h1 { font-size: 18px; font-weight: 600; }
    .header h1 span { color: var(--accent); }
    .header .status { margin-left: auto; font-size: 12px; color: var(--green); }
    .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 12px 0; overflow-y: auto; }
    .sidebar button { display: block; width: 100%; padding: 10px 20px; text-align: left; background: none;
                      border: none; color: var(--text-dim); font-size: 14px; cursor: pointer; }
    .sidebar button:hover { background: rgba(88,166,255,0.1); color: var(--text); }
    .sidebar button.active { color: var(--accent); background: rgba(88,166,255,0.1);
                             border-left: 2px solid var(--accent); }
    .main { overflow-y: auto; padding: 24px; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
    .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: var(--text); }
    .card h3 { font-size: 14px; font-weight: 500; margin-bottom: 8px; color: var(--text-dim); }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
                 padding: 16px; text-align: center; }
    .stat-card .value { font-size: 32px; font-weight: 700; color: var(--accent); }
    .stat-card .label { font-size: 12px; color: var(--text-dim); margin-top: 4px; }
    input, textarea { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                      color: var(--text); padding: 8px 12px; font-size: 14px; width: 100%; }
    input:focus, textarea:focus { outline: none; border-color: var(--accent); }
    button.primary { background: var(--accent); color: #fff; border: none; border-radius: 6px;
                     padding: 8px 16px; font-size: 14px; cursor: pointer; font-weight: 500; }
    button.primary:hover { opacity: 0.9; }
    button.secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border);
                       border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; }
    .relation-row { display: flex; align-items: center; gap: 8px; padding: 6px 0;
                    border-bottom: 1px solid var(--border); font-size: 13px; }
    .relation-row .subj { color: var(--green); font-weight: 500; }
    .relation-row .pred { color: var(--yellow); padding: 2px 6px; background: rgba(210,153,34,0.1);
                          border-radius: 4px; font-size: 12px; }
    .relation-row .obj { color: var(--cyan); font-weight: 500; }
    .relation-row .tag { color: var(--purple); font-size: 11px; }
    .answer-box { background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3);
                  border-radius: 6px; padding: 12px 16px; margin-top: 8px; }
    .answer-box .answer { font-size: 16px; font-weight: 600; color: var(--green); }
    .answer-box .meta { font-size: 12px; color: var(--text-dim); margin-top: 4px; }
    .trace-span { padding: 4px 8px; margin: 2px 0; border-left: 3px solid var(--accent);
                  background: rgba(88,166,255,0.05); font-size: 13px; font-family: monospace; }
    .trace-span .name { color: var(--accent); font-weight: 500; }
    .trace-span .dur { color: var(--yellow); margin-left: 8px; }
    svg { width: 100%; }
    .node-label { font-size: 11px; fill: var(--text); pointer-events: none; }
    .link-label { font-size: 9px; fill: var(--text-dim); }
    .tooltip { position: absolute; background: var(--surface); border: 1px solid var(--border);
               border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 100; }
    .metrics-chart { height: 200px; position: relative; }
    .metrics-chart canvas { width: 100% !important; height: 100% !important; }
    .tab-bar { display: flex; gap: 4px; margin-bottom: 16px; }
    .tab-bar button { padding: 6px 12px; border-radius: 6px; }
  </style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const { useState, useEffect, useRef, useCallback } = React;

// ─── API Client ──────────────────────────────────────────────
const api = {
  get: (path) => fetch(`/api/v1${path}`).then(r => r.json()),
  post: (path, body) => fetch(`/api/v1${path}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(r => r.json()),
};

// ─── Hooks ───────────────────────────────────────────────────
function usePoll(fetcher, interval = 5000) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const d = await fetcher();
        if (active) setData(d);
      } catch (e) { if (active) setError(e); }
    };
    load();
    const id = setInterval(load, interval);
    return () => { active = false; clearInterval(id); };
  }, []);
  return { data, error };
}

// ─── Graph Visualizer (D3) ───────────────────────────────────
function GraphVisualizer({ onEntityClick }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [graphData, setGraphData] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    api.get('/graph/network').then(setGraphData);
  }, []);

  useEffect(() => {
    if (!graphData || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth;
    const height = 500;
    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().on('zoom', (e) => g.attr('transform', e.transform)));

    const colorMap = { person: '#3fb950', company: '#58a6ff', place: '#f0883e', concept: '#bc8cff', unknown: '#8b949e' };

    // Arrow markers
    svg.append('defs').selectAll('marker')
      .data(['arrow']).enter().append('marker')
      .attr('id', 'arrow').attr('viewBox', '0 -5 10 10')
      .attr('refX', 20).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto').append('path').attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#30363d');

    const sim = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width/2, height/2))
      .force('collision', d3.forceCollide(30));

    const link = g.selectAll('.link').data(graphData.links).enter().append('line')
      .attr('stroke', d => d.inferred ? '#30363d' : '#484f58')
      .attr('stroke-width', d => d.inferred ? 1 : 2)
      .attr('stroke-dasharray', d => d.inferred ? '4,4' : 'none')
      .attr('marker-end', 'url(#arrow)');

    const linkLabel = g.selectAll('.link-label').data(graphData.links).enter().append('text')
      .attr('class', 'link-label').attr('text-anchor', 'middle')
      .text(d => d.label);

    const node = g.selectAll('.node').data(graphData.nodes).enter().append('circle')
      .attr('r', 10).attr('fill', d => colorMap[d.type] || colorMap.unknown)
      .attr('stroke', '#0d1117').attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (e, d) => onEntityClick && onEntityClick(d.label))
      .on('mouseover', (e, d) => setTooltip({ x: e.pageX, y: e.pageY, text: `${d.label} (${d.type})` }))
      .on('mouseout', () => setTooltip(null))
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    const label = g.selectAll('.node-label').data(graphData.nodes).enter().append('text')
      .attr('class', 'node-label').attr('dx', 14).attr('dy', 4)
      .text(d => d.label);

    sim.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      linkLabel.attr('x', d => (d.source.x + d.target.x)/2).attr('y', d => (d.source.y + d.target.y)/2);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y);
    });

    simRef.current = sim;
    return () => sim.stop();
  }, [graphData]);

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={svgRef} height="500" style={{ background: 'var(--bg)', borderRadius: '8px' }} />
      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x + 10, top: tooltip.y - 30 }}>
          {tooltip.text}
        </div>
      )}
      {graphData && (
        <div style={{ position: 'absolute', top: 8, right: 12, fontSize: 12, color: 'var(--text-dim)' }}>
          {graphData.stats.node_count} nodes · {graphData.stats.edge_count} edges
        </div>
      )}
    </div>
  );
}

// ─── Dashboard View ──────────────────────────────────────────
function DashboardView() {
  const { data: stats } = usePoll(() => api.get('/stats'), 3000);
  const { data: traces } = usePoll(() => api.get('/telemetry/traces?limit=10'), 5000);

  return (
    <>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="value">{stats?.entities ?? '—'}</div>
          <div className="label">Entities</div>
        </div>
        <div className="stat-card">
          <div className="value">{stats?.relations ?? '—'}</div>
          <div className="label">Relations</div>
        </div>
        <div className="stat-card">
          <div className="value">{stats?.explicit_relations ?? '—'}</div>
          <div className="label">Explicit</div>
        </div>
        <div className="stat-card">
          <div className="value">{stats?.inferred_relations ?? '—'}</div>
          <div className="label">Inferred</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Recent Traces</h2>
        {traces?.traces?.length === 0 && <p style={{ color: 'var(--text-dim)' }}>No traces yet. Run a query to generate traces.</p>}
        {traces?.traces?.map(t => (
          <div key={t.trace_id} className="trace-span" style={{ marginBottom: 4 }}>
            <span className="name">{t.root_span}</span>
            <span className="dur">{(t.duration_ms / 1e6).toFixed(1)}ms</span>
            <span style={{ color: 'var(--text-dim)', marginLeft: 8 }}>{t.span_count} spans</span>
          </div>
        ))}
      </div>
    </>
  );
}

// ─── Relations View ──────────────────────────────────────────
function RelationsView() {
  const [rels, setRels] = useState(null);
  useEffect(() => { api.get('/graph/relations').then(setRels); }, []);
  return (
    <div className="card">
      <h2>Knowledge Graph Relations ({rels?.count ?? 0})</h2>
      <p style={{ color: 'var(--text-dim)', marginBottom: 12, fontSize: 13 }}>
        {rels?.explicit ?? 0} explicit · {rels?.inferred ?? 0} inferred
      </p>
      {rels?.relations?.map((r, i) => (
        <div key={i} className="relation-row">
          <span className="subj">{r.subject_name}</span>
          <span className="pred">{r.predicate}</span>
          <span className="obj">{r.object_name}</span>
          {r.is_inferred && <span className="tag">[inferred]</span>}
          <span style={{ marginLeft: 'auto', color: 'var(--text-dim)', fontSize: 11 }}>
            {(r.confidence * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Entity Explorer ─────────────────────────────────────────
function EntityExplorer({ initialEntity }) {
  const [entities, setEntities] = useState(null);
  const [selected, setSelected] = useState(initialEntity || null);
  const [detail, setDetail] = useState(null);

  useEffect(() => { api.get('/graph/entities').then(setEntities); }, []);
  useEffect(() => {
    if (selected) api.get(`/graph/entity/${encodeURIComponent(selected)}`).then(setDetail);
    else setDetail(null);
  }, [selected]);
  useEffect(() => { if (initialEntity) setSelected(initialEntity); }, [initialEntity]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: 16 }}>
      <div className="card">
        <h2>Entities ({entities?.count ?? 0})</h2>
        {entities?.entities?.map(e => (
          <div key={e.id} onClick={() => setSelected(e.name)}
               style={{ padding: '8px 12px', cursor: 'pointer', borderRadius: 6,
                        background: selected === e.name ? 'rgba(88,166,255,0.1)' : 'transparent',
                        borderBottom: '1px solid var(--border)' }}>
            <div style={{ fontWeight: 500 }}>{e.name}</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>{e.type} · {e.relation_count} relations</div>
          </div>
        ))}
      </div>
      <div className="card">
        {!detail ? <p style={{ color: 'var(--text-dim)' }}>Select an entity</p> : (
          <>
            <h2>{detail.entity.name}</h2>
            <p style={{ color: 'var(--text-dim)', marginBottom: 12 }}>Type: {detail.entity.type}</p>
            <h3>Outgoing Relations ({detail.outgoing.length})</h3>
            {detail.outgoing.map((r, i) => (
              <div key={i} className="relation-row">
                <span className="subj">{r.subject_name}</span>
                <span className="pred">{r.predicate}</span>
                <span className="obj" style={{ cursor: 'pointer' }} onClick={() => setSelected(r.object_name)}>{r.object_name}</span>
                {r.is_inferred && <span className="tag">[inferred]</span>}
              </div>
            ))}
            <h3 style={{ marginTop: 16 }}>Incoming Relations ({detail.incoming.length})</h3>
            {detail.incoming.map((r, i) => (
              <div key={i} className="relation-row">
                <span className="subj" style={{ cursor: 'pointer' }} onClick={() => setSelected(r.subject_name)}>{r.subject_name}</span>
                <span className="pred">{r.predicate}</span>
                <span className="obj">{r.object_name}</span>
                {r.is_inferred && <span className="tag">[inferred]</span>}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ─── Chat View ──────────────────────────────────────────────
function ChatView() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const res = await api.post('/chat', { message: userMsg });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.response,
        meta: {
          llm_used: res.llm_used,
          search_used: res.search_used,
          search_results_count: res.search_results_count || 0,
          graph_answered: res.graph_answered,
          graph_confidence: res.graph_confidence,
          reason_answer: res.reason_answer,
          reason_confidence: res.reason_confidence,
          memories_retrieved: res.memories_retrieved,
          signals_fired: res.signals_fired || [],
          duration_ms: res.duration_ms,
          context: res.context,
        },
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'error', content: e.message }]);
    }
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 56px - 48px)' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 0 16px 0' }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', marginTop: 80, color: 'var(--text-dim)' }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>&#129504;</div>
            <div style={{ fontSize: 16, fontWeight: 500 }}>Limbiq Chat</div>
            <div style={{ fontSize: 13, marginTop: 4 }}>Tell me things. Ask me questions. I learn as we talk.</div>
            <div style={{ fontSize: 12, marginTop: 16, color: 'var(--text-dim)' }}>
              Try: "My name is Dimuthu" then "My wife is Prabhashi" then "What is my wife's name?"
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            {m.role === 'user' && (
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{ background: 'var(--accent)', color: '#fff', padding: '10px 14px',
                              borderRadius: '16px 16px 4px 16px', maxWidth: '70%', fontSize: 14 }}>
                  {m.content}
                </div>
              </div>
            )}
            {m.role === 'assistant' && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 4 }}>
                <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', padding: '10px 14px',
                              borderRadius: '16px 16px 16px 4px', maxWidth: '80%', fontSize: 14 }}>
                  {m.content}
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-dim)', paddingLeft: 4 }}>
                  {m.meta?.llm_used && (
                    <span style={{ color: 'var(--cyan)' }}>LLM</span>
                  )}
                  {m.meta?.search_used && (
                    <span style={{ color: 'var(--orange)' }}>
                      search ({m.meta.search_results_count})
                    </span>
                  )}
                  {m.meta?.graph_answered && (
                    <span style={{ color: 'var(--green)' }}>
                      graph ({(m.meta.graph_confidence * 100).toFixed(0)}%)
                    </span>
                  )}
                  {m.meta?.reason_answer && (
                    <span style={{ color: 'var(--purple)' }}>
                      reasoner ({(m.meta.reason_confidence * 100).toFixed(0)}%)
                    </span>
                  )}
                  {m.meta?.memories_retrieved > 0 && (
                    <span>{m.meta.memories_retrieved} memories</span>
                  )}
                  {m.meta?.signals_fired?.length > 0 && (
                    <span style={{ color: 'var(--yellow)' }}>
                      {m.meta.signals_fired.map(s => s.type).join(', ')}
                    </span>
                  )}
                  <span>{m.meta?.duration_ms?.toFixed(0)}ms</span>
                </div>
                {m.meta?.context && (
                  <details style={{ fontSize: 11, color: 'var(--text-dim)', paddingLeft: 4, width: '80%' }}>
                    <summary style={{ cursor: 'pointer' }}>memory context</summary>
                    <pre style={{ marginTop: 4, whiteSpace: 'pre-wrap', fontSize: 11, background: 'var(--bg)',
                                  padding: 8, borderRadius: 6, maxHeight: 150, overflow: 'auto' }}>
                      {m.meta.context}
                    </pre>
                  </details>
                )}
              </div>
            )}
            {m.role === 'error' && (
              <div style={{ color: 'var(--red)', fontSize: 13, padding: '8px 14px' }}>
                Error: {m.content}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', padding: '10px 14px',
                          borderRadius: '16px 16px 16px 4px', fontSize: 14, color: 'var(--text-dim)' }}>
              Thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div style={{ borderTop: '1px solid var(--border)', padding: '12px 0 0 0', display: 'flex', gap: 8 }}>
        <input value={input} onChange={e => setInput(e.target.value)}
               placeholder="Tell me something or ask a question..."
               onKeyDown={e => e.key === 'Enter' && send()}
               style={{ flex: 1 }} />
        <button className="primary" onClick={send} disabled={loading}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

// ─── Query Builder ───────────────────────────────────────────
function QueryView() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!query.trim()) return;
    setLoading(true);
    const body = { message: query };
    const [graphR, processR, reasonR] = await Promise.all([
      api.post('/graph/query', body).catch(e => ({ error: e.message })),
      api.post('/process', body).catch(e => ({ error: e.message })),
      api.post('/reason', body).catch(e => ({ error: e.message })),
    ]);
    setResults([
      { title: 'Graph Query', data: graphR },
      { title: 'Process (Memory)', data: processR },
      { title: 'Reasoner (Phase 5)', data: reasonR },
    ]);
    setLoading(false);
  };

  return (
    <>
      <div className="card">
        <h2>Query Builder</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <input value={query} onChange={e => setQuery(e.target.value)}
                 placeholder="Ask a question... e.g. 'Who is my father's wife?'"
                 onKeyDown={e => e.key === 'Enter' && run()} />
          <button className="primary" onClick={run} disabled={loading}>
            {loading ? '...' : 'Query'}
          </button>
        </div>
      </div>
      {results.map((r, i) => (
        <div key={i} className="card">
          <h2>{r.title}</h2>
          {r.data?.answered && (
            <div className="answer-box">
              <div className="answer">{r.data.answer}</div>
              <div className="meta">
                Confidence: {((r.data.confidence || 0) * 100).toFixed(0)}%
                {r.data.duration_ms && ` · ${r.data.duration_ms.toFixed(1)}ms`}
                {r.data.answer_mode && ` · Mode: ${r.data.answer_mode}`}
                {r.data.reasoning_trace && ` · ${r.data.reasoning_trace}`}
              </div>
            </div>
          )}
          {!r.data?.answered && r.data?.context && (
            <details>
              <summary style={{ cursor: 'pointer', color: 'var(--text-dim)' }}>Memory Context ({r.data.memories_retrieved} memories)</summary>
              <pre style={{ marginTop: 8, fontSize: 12, color: 'var(--text-dim)', whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
                {r.data.context}
              </pre>
            </details>
          )}
          {!r.data?.answered && !r.data?.context && (
            <p style={{ color: 'var(--text-dim)' }}>No answer · {r.data?.duration_ms?.toFixed(1) || '?'}ms</p>
          )}
          {r.data?.error && <p style={{ color: 'var(--red)' }}>Error: {r.data.error}</p>}
        </div>
      ))}
    </>
  );
}

// ─── Traces View ─────────────────────────────────────────────
function TracesView() {
  const { data } = usePoll(() => api.get('/telemetry/traces?limit=30'), 3000);
  const [expanded, setExpanded] = useState(null);

  return (
    <div className="card">
      <h2>OpenTelemetry Traces</h2>
      {!data?.traces?.length && <p style={{ color: 'var(--text-dim)' }}>No traces captured yet. Run queries to generate traces.</p>}
      {data?.traces?.map(t => (
        <div key={t.trace_id} style={{ marginBottom: 8 }}>
          <div onClick={() => setExpanded(expanded === t.trace_id ? null : t.trace_id)}
               style={{ cursor: 'pointer', padding: 8, background: 'var(--bg)', borderRadius: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 500 }}>{t.root_span}</span>
            <span style={{ color: 'var(--yellow)', fontSize: 12 }}>{(t.duration_ms / 1e6).toFixed(1)}ms</span>
            <span style={{ color: 'var(--text-dim)', fontSize: 12, marginLeft: 'auto' }}>{t.span_count} spans</span>
          </div>
          {expanded === t.trace_id && (
            <div style={{ paddingLeft: 16, marginTop: 4 }}>
              {t.spans.map((s, i) => (
                <div key={i} className="trace-span" style={{ marginLeft: s.parent_span_id ? 20 : 0 }}>
                  <span className="name">{s.name}</span>
                  <span className="dur">{s.duration_ms.toFixed(2)}ms</span>
                  {Object.keys(s.attributes).length > 0 && (
                    <details style={{ marginTop: 4 }}>
                      <summary style={{ fontSize: 11, color: 'var(--text-dim)', cursor: 'pointer' }}>attributes</summary>
                      <pre style={{ fontSize: 11, color: 'var(--text-dim)' }}>{JSON.stringify(s.attributes, null, 2)}</pre>
                    </details>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────
function App() {
  const [view, setView] = useState('chat');
  const [selectedEntity, setSelectedEntity] = useState(null);

  const handleEntityClick = (name) => { setSelectedEntity(name); setView('entities'); };

  const views = {
    chat: () => <ChatView />,
    dashboard: () => <DashboardView />,
    graph: () => <GraphVisualizer onEntityClick={handleEntityClick} />,
    entities: () => <EntityExplorer initialEntity={selectedEntity} />,
    relations: () => <RelationsView />,
    query: () => <QueryView />,
    traces: () => <TracesView />,
  };

  return (
    <div className="app">
      <div className="header">
        <h1><span>limbiq</span> playground</h1>
        <div className="status">● connected</div>
      </div>
      <div className="sidebar">
        {[
          ['chat', '◉ Chat'],
          ['dashboard', '◎ Dashboard'],
          ['graph', '◈ Knowledge Graph'],
          ['entities', '◇ Entity Explorer'],
          ['relations', '▷ Relations'],
          ['query', '⊡ Query Builder'],
          ['traces', '⊞ Traces'],
        ].map(([key, label]) => (
          <button key={key} className={view === key ? 'active' : ''}
                  onClick={() => { setView(key); if (key !== 'entities') setSelectedEntity(null); }}>
            {label}
          </button>
        ))}
      </div>
      <div className="main">
        {views[view]?.()}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>
</body>
</html>"""
