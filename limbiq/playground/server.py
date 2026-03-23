"""
FastAPI server for Limbiq Playground.

Creates the app, mounts API routes, and serves an inline SPA dashboard.
Focused on: signals + graph generation + self-healing connectivity.
"""

import os
import time
import logging
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
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup/shutdown lifecycle."""
        from limbiq import Limbiq

        logger.info(f"Initializing Limbiq (store={store_path}, user={user_id})")
        lq = Limbiq(
            store_path=store_path,
            user_id=user_id,
            embedding_model=embedding_model,
            llm_fn=llm_client,
        )
        app.state.lq = lq
        app.state.llm = llm_client
        app.state.start_time = time.time()

        lq.start_session()
        logger.info("Limbiq Playground ready")

        yield

        # Shutdown
        try:
            lq.end_session()
        except Exception as e:
            logger.warning(f"End session failed during shutdown: {e}")

    app = FastAPI(
        title="Limbiq Playground",
        description="Signals + Graph dashboard",
        version="0.6.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8765", "http://127.0.0.1:8765"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return _get_dashboard_html()

    return app


def _get_dashboard_html() -> str:
    """Return the inline SPA dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Limbiq Playground</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --border: #2d3148;
    --text: #e1e4ed;
    --text2: #8b8fa3;
    --accent: #6c5ce7;
    --accent2: #a29bfe;
    --green: #00b894;
    --red: #e17055;
    --yellow: #fdcb6e;
    --blue: #74b9ff;
    --cyan: #81ecec;
    --pink: #fd79a8;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    overflow: hidden;
    height: 100vh;
  }
  .layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 48px 1fr 280px;
    height: 100vh;
    gap: 1px;
    background: var(--border);
  }
  .header {
    grid-column: 1 / -1;
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
  }
  .header h1 { font-size: 15px; font-weight: 600; }
  .header .stats { display: flex; gap: 16px; color: var(--text2); font-size: 12px; }
  .header .stat { display: flex; align-items: center; gap: 4px; }
  .header .dot { width: 8px; height: 8px; border-radius: 50%; }
  .header .dot.green { background: var(--green); }
  .header .dot.red { background: var(--red); }
  .header .dot.yellow { background: var(--yellow); }

  .panel {
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-header {
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }

  /* Chat panel */
  .chat-messages { flex: 1; overflow-y: auto; padding: 8px; }
  .msg { margin-bottom: 8px; padding: 8px 10px; border-radius: 6px; line-height: 1.5; }
  .msg.user { background: var(--surface2); border-left: 3px solid var(--accent); }
  .msg.assistant { background: var(--surface2); border-left: 3px solid var(--green); }
  .msg.system { background: var(--surface2); border-left: 3px solid var(--yellow); font-size: 11px; color: var(--text2); }
  .msg .label { font-size: 10px; color: var(--text2); margin-bottom: 2px; text-transform: uppercase; }

  .chat-input {
    display: flex;
    gap: 8px;
    padding: 8px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  .chat-input input {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
    outline: none;
  }
  .chat-input input:focus { border-color: var(--accent); }
  .chat-input button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-weight: 600;
  }
  .chat-input button:hover { background: var(--accent2); }
  .chat-input button:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Graph panel */
  #graph-svg { width: 100%; height: 100%; }
  .node circle { stroke: var(--border); stroke-width: 1.5; }
  .node text { fill: var(--text); font-size: 10px; pointer-events: none; }
  .link { stroke-opacity: 0.6; }
  .link-label { fill: var(--text2); font-size: 8px; pointer-events: none; }

  /* Signal log */
  .signal-item {
    padding: 6px 8px;
    margin-bottom: 4px;
    border-radius: 4px;
    background: var(--surface2);
    font-size: 11px;
    display: flex;
    justify-content: space-between;
  }
  .signal-type {
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
  }
  .signal-type.dopamine { background: #6c5ce733; color: var(--accent2); }
  .signal-type.gaba { background: #e1705533; color: var(--red); }
  .signal-type.serotonin { background: #00b89433; color: var(--green); }
  .signal-type.acetylcholine { background: #74b9ff33; color: var(--blue); }
  .signal-type.norepinephrine { background: #fdcb6e33; color: var(--yellow); }

  /* Connectivity */
  .connectivity {
    display: flex;
    gap: 12px;
    padding: 8px 12px;
    font-size: 12px;
    align-items: center;
  }
  .connectivity .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
  }
  .connectivity .badge.connected { background: #00b89433; color: var(--green); }
  .connectivity .badge.disconnected { background: #e1705533; color: var(--red); }

  /* Bottom panels */
  .bottom-panels {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
  }

  /* Buttons */
  .btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text2);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 10px;
    font-family: inherit;
  }
  .btn:hover { background: var(--border); color: var(--text); }

  /* Entity list */
  .entity-item {
    padding: 4px 8px;
    margin-bottom: 2px;
    border-radius: 3px;
    background: var(--surface2);
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .entity-type {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    background: var(--border);
    color: var(--text2);
  }

  .scrollbar::-webkit-scrollbar { width: 6px; }
  .scrollbar::-webkit-scrollbar-track { background: transparent; }
  .scrollbar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="layout">
  <!-- Header -->
  <div class="header">
    <h1>Limbiq Playground</h1>
    <div class="stats">
      <div class="stat" id="conn-badge">
        <div class="dot green" id="conn-dot"></div>
        <span id="conn-text">Loading...</span>
      </div>
      <div class="stat">Entities: <span id="stat-entities">0</span></div>
      <div class="stat">Relations: <span id="stat-relations">0</span></div>
      <div class="stat">Signals: <span id="stat-signals">0</span></div>
    </div>
  </div>

  <!-- Chat panel -->
  <div class="panel">
    <div class="panel-header">
      Chat
      <div>
        <button class="btn" onclick="startSession()">New Session</button>
        <button class="btn" onclick="endSession()">End Session</button>
      </div>
    </div>
    <div class="chat-messages scrollbar" id="chat-messages"></div>
    <div class="chat-input">
      <input type="text" id="chat-input" placeholder="Type a message..."
             onkeydown="if(event.key==='Enter')sendMessage()">
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <!-- Graph panel -->
  <div class="panel">
    <div class="panel-header">
      Knowledge Graph
      <div>
        <button class="btn" onclick="healGraph()">Heal</button>
        <button class="btn" onclick="refreshGraph()">Refresh</button>
      </div>
    </div>
    <div class="panel-body" id="graph-container" style="padding: 0;">
      <svg id="graph-svg"></svg>
    </div>
  </div>

  <!-- Bottom panels -->
  <div class="bottom-panels">
    <!-- Signals -->
    <div class="panel">
      <div class="panel-header">
        Signal Log
        <button class="btn" onclick="refreshSignals()">Refresh</button>
      </div>
      <div class="panel-body scrollbar" id="signal-log"></div>
    </div>

    <!-- Entities -->
    <div class="panel">
      <div class="panel-header">
        Entities
        <button class="btn" onclick="refreshEntities()">Refresh</button>
      </div>
      <div class="panel-body scrollbar" id="entity-list"></div>
    </div>

    <!-- World Summary / Memory -->
    <div class="panel">
      <div class="panel-header">
        World Summary
        <button class="btn" onclick="refreshProfile()">Refresh</button>
      </div>
      <div class="panel-body scrollbar" id="world-summary" style="line-height: 1.6;"></div>
    </div>
  </div>
</div>

<script>
const API = '/api/v1';
let signalCount = 0;

// ── API helpers ──────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return res.json();
}

// ── Chat ─────────────────────────────────────────────────────

function addMessage(role, content, extra = '') {
  const div = document.getElementById('chat-messages');
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  msg.innerHTML = `<div class="label">${role}${extra ? ' · ' + extra : ''}</div>${escHtml(content)}`;
  div.appendChild(msg);
  div.scrollTop = div.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('send-btn');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  btn.disabled = true;
  addMessage('user', msg);

  try {
    const data = await api('/chat', {
      method: 'POST',
      body: JSON.stringify({ message: msg }),
    });

    addMessage('assistant', data.response || '[no response]',
      `${data.duration_ms}ms · ${data.memories_retrieved} memories`);

    // Show signals
    if (data.signals && data.signals.length > 0) {
      const sigText = data.signals.map(s => `${s.signal_type}: ${s.trigger}`).join(', ');
      addMessage('system', `Signals fired: ${sigText}`);
      signalCount += data.signals.length;
    }

    // Update connectivity badge
    if (data.connectivity) updateConnectivity(data.connectivity);

    // Refresh panels
    refreshGraph();
    refreshSignals();
    refreshEntities();
    refreshProfile();
    updateStats();
  } catch (e) {
    addMessage('system', 'Error: ' + e.message);
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

// ── Sessions ─────────────────────────────────────────────────

async function startSession() {
  await api('/session/start', { method: 'POST' });
  addMessage('system', 'New session started');
}

async function endSession() {
  const data = await api('/session/end', { method: 'POST' });
  addMessage('system', `Session ended (${data.duration_ms}ms, inferred: ${data.graph_inferred || 0})`);
  refreshGraph();
  refreshEntities();
  refreshProfile();
}

// ── Graph ─────────────────────────────────────────────────────

let simulation = null;
let nodePositions = {};  // Persist positions across refreshes: { id: {x, y} }
let graphInitialized = false;

function refreshGraph() {
  api('/graph/network').then(data => {
    renderGraph(data.nodes, data.links);
    document.getElementById('stat-entities').textContent = data.nodes.length;
    document.getElementById('stat-relations').textContent = data.links.length;
  });
}

const typeColors = {
  person: '#6c5ce7', place: '#00b894', company: '#fdcb6e',
  animal: '#fd79a8', concept: '#74b9ff', condition: '#e17055',
  unknown: '#636e72',
};

function renderGraph(nodes, links) {
  const container = document.getElementById('graph-container');
  const svg = d3.select('#graph-svg');
  const w = container.clientWidth;
  const h = container.clientHeight;
  svg.attr('width', w).attr('height', h);

  if (nodes.length === 0) {
    svg.selectAll('*').remove();
    svg.append('text').attr('x', w/2).attr('y', h/2)
      .attr('text-anchor', 'middle').attr('fill', '#8b8fa3')
      .text('No entities yet. Chat to build the graph.');
    if (simulation) { simulation.stop(); simulation = null; }
    graphInitialized = false;
    return;
  }

  // Restore saved positions for existing nodes; place new nodes near center
  nodes.forEach(n => {
    if (nodePositions[n.id]) {
      n.x = nodePositions[n.id].x;
      n.y = nodePositions[n.id].y;
      n.vx = 0;
      n.vy = 0;
      // Re-pin nodes that were previously dragged by the user
      if (nodePositions[n.id].pinned) {
        n.fx = n.x;
        n.fy = n.y;
      }
    } else {
      // Spread new nodes around center so they don't stack
      n.x = w / 2 + (Math.random() - 0.5) * 100;
      n.y = h / 2 + (Math.random() - 0.5) * 100;
    }
  });

  // Initialize SVG groups once, reuse via D3 data joins
  if (!graphInitialized) {
    svg.selectAll('*').remove();
    svg.append('g').attr('class', 'links-group');
    svg.append('g').attr('class', 'link-labels-group');
    svg.append('g').attr('class', 'nodes-group');
    graphInitialized = true;
  }

  // --- Data join for links ---
  const linkSel = svg.select('.links-group')
    .selectAll('line')
    .data(links, d => d.source_id + '-' + d.predicate + '-' + d.target_id);
  linkSel.exit().remove();
  const linkEnter = linkSel.enter().append('line').attr('class', 'link');
  const link = linkEnter.merge(linkSel)
    .attr('stroke', d => d.is_inferred ? '#4a4d5e' : '#6c5ce755')
    .attr('stroke-width', d => d.is_inferred ? 1 : 2)
    .attr('stroke-dasharray', d => d.is_inferred ? '4,4' : 'none');

  // --- Data join for link labels ---
  const labelSel = svg.select('.link-labels-group')
    .selectAll('text')
    .data(links, d => d.source_id + '-' + d.predicate + '-' + d.target_id);
  labelSel.exit().remove();
  const labelEnter = labelSel.enter().append('text').attr('class', 'link-label');
  const linkLabel = labelEnter.merge(labelSel).text(d => d.predicate);

  // --- Data join for nodes ---
  const nodeSel = svg.select('.nodes-group')
    .selectAll('g.node')
    .data(nodes, d => d.id);
  nodeSel.exit().remove();
  const nodeEnter = nodeSel.enter().append('g').attr('class', 'node');
  nodeEnter.append('circle');
  nodeEnter.append('text').attr('dx', 12).attr('dy', 4);
  const node = nodeEnter.merge(nodeSel);

  node.select('circle')
    .attr('r', d => 6 + Math.min(d.relations * 2, 12))
    .attr('fill', d => typeColors[d.type] || typeColors.unknown);
  node.select('text').text(d => d.name);

  // Drag behavior — nodes stay pinned where you drop them
  // Double-click a node to unpin it
  node.call(d3.drag()
    .on('start', (e, d) => {
      if (!e.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => {
      if (!e.active) simulation.alphaTarget(0);
      // Keep node pinned where user dropped it
      d.fx = e.x; d.fy = e.y;
      nodePositions[d.id] = { x: d.x, y: d.y, pinned: true };
    })
  );
  // Double-click to unpin
  node.on('dblclick', (e, d) => {
    d.fx = null; d.fy = null;
    if (nodePositions[d.id]) nodePositions[d.id].pinned = false;
    simulation.alpha(0.1).restart();
  });

  // --- Simulation: reuse or create with low alpha for stability ---
  if (simulation) {
    simulation.nodes(nodes);
    simulation.force('link').links(links);
    // Gentle reheat — just enough for new nodes to settle
    simulation.alpha(0.15).restart();
  } else {
    simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX(w / 2).strength(0.03))
      .force('y', d3.forceY(h / 2).strength(0.03))
      .force('collision', d3.forceCollide().radius(30))
      .velocityDecay(0.4)
      .alpha(0.5);
  }

  simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    linkLabel.attr('x', d => (d.source.x + d.target.x) / 2)
             .attr('y', d => (d.source.y + d.target.y) / 2);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
    // Continuously save positions
    nodes.forEach(n => { nodePositions[n.id] = { x: n.x, y: n.y }; });
  });
}

async function healGraph() {
  const data = await api('/graph/heal', { method: 'POST' });
  addMessage('system', `Graph healed (${data.duration_ms}ms)`);
  if (data.connectivity) updateConnectivity(data.connectivity);
  refreshGraph();
  refreshEntities();
}

// ── Connectivity ─────────────────────────────────────────────

function updateConnectivity(conn) {
  const dot = document.getElementById('conn-dot');
  const text = document.getElementById('conn-text');
  if (conn.fully_connected) {
    dot.className = 'dot green';
    text.textContent = `Connected (${conn.entities} nodes)`;
  } else {
    dot.className = 'dot red';
    text.textContent = `${conn.components} components`;
  }
}

// ── Signals ──────────────────────────────────────────────────

function refreshSignals() {
  api('/signals/log?limit=30').then(events => {
    const div = document.getElementById('signal-log');
    div.innerHTML = events.map(e => `
      <div class="signal-item">
        <span class="signal-type ${e.signal_type}">${e.signal_type}</span>
        <span>${e.trigger}</span>
      </div>
    `).join('');
    document.getElementById('stat-signals').textContent = events.length;
  });
}

// ── Entities ─────────────────────────────────────────────────

function refreshEntities() {
  api('/graph/entities').then(entities => {
    const div = document.getElementById('entity-list');
    div.innerHTML = entities.map(e => `
      <div class="entity-item">
        <span>${e.name}</span>
        <span class="entity-type">${e.type} · ${e.relation_count} rels</span>
      </div>
    `).join('');
  });
}

// ── World Summary ────────────────────────────────────────────

function refreshProfile() {
  api('/profile').then(data => {
    const div = document.getElementById('world-summary');
    let html = '';
    if (data.world_summary) {
      html += `<div style="margin-bottom:8px;color:var(--accent2);">${escHtml(data.world_summary)}</div>`;
    }
    if (data.priority_facts && data.priority_facts.length > 0) {
      html += '<div style="color:var(--text2);margin-top:6px;">Priority facts:</div>';
      data.priority_facts.forEach(f => {
        html += `<div style="padding:2px 0;">&bull; ${escHtml(f)}</div>`;
      });
    }
    if (!html) html = '<div style="color:var(--text2);">No data yet. Chat to build knowledge.</div>';
    div.innerHTML = html;
  });
}

// ── Stats ────────────────────────────────────────────────────

function updateStats() {
  api('/graph/connectivity').then(updateConnectivity);
}

// ── Init ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  refreshGraph();
  refreshSignals();
  refreshEntities();
  refreshProfile();
  updateStats();
  document.getElementById('chat-input').focus();
});
</script>
</body>
</html>"""
