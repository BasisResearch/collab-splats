import { state, registerTab, setProgress } from './state.js';

let currentFrameIdx = 0;
let totalFrames = 0;
let isQueryable = false;

// ── Sidebar ───────────────────────────────────────────────────────
function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Semantic methods</h4>
    <label>Extractor</label>
    <select id="sem-extractor"><option value="">— loading… —</option></select>
    <div id="sem-query-section" style="display:none;flex-direction:column;gap:4px;margin-top:6px">
      <label>Positive query</label>
      <input type="text" id="sem-query-pos" placeholder="e.g. bird" value="bird">
      <label>Negative query</label>
      <input type="text" id="sem-query-neg" placeholder="background, ground, sky" value="background, ground, sky">
      <button class="primary" id="sem-query-btn">&#9654; Query frame</button>
    </div>
    <button class="primary" id="sem-run-btn" style="margin-top:8px">&#9654; Extract features</button>
    <div id="sem-status" class="status-info"></div>
  `;

  // Load methods + cached status; prefer cached extractors
  Promise.all([
    fetch('/api/semantics/methods').then(r => r.json()).catch(() => ({methods: ['dinov2']})),
    fetch('/api/semantics/status').then(r => r.json()).catch(() => ({cached: []})),
  ]).then(([methodsData, statusData]) => {
    const sel = sec.querySelector('#sem-extractor');
    const cached = new Set(statusData.cached || []);
    const methods = methodsData.methods || ['dinov2'];
    sel.innerHTML = '';
    const ordered = [...methods.filter(m => cached.has(m)), ...methods.filter(m => !cached.has(m))];
    ordered.forEach(m => {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = cached.has(m) ? `${m} ✓` : m;
      if (m === state.extractor || (cached.size > 0 && cached.has(m))) o.selected = true;
      sel.appendChild(o);
    });
    const status = sec.querySelector('#sem-status');
    if (status && cached.size > 0) { status.textContent = `✓ Cached: ${[...cached].join(', ')}`; status.className = 'status-ok'; }
    if (cached.size > 0) {
      const first = ordered[0];
      sel.value = first;
      state.extractor = first;
      fetch('/api/session/update', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({extractor: first}) });
      checkQueryable(first, sec);
    }
  });

  sec.querySelector('#sem-extractor').addEventListener('change', e => {
    state.extractor = e.target.value.replace(' ✓', '');
    fetch('/api/session/update', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({extractor: state.extractor}) });
    checkQueryable(state.extractor, sec);
  });
  sec.querySelector('#sem-run-btn').addEventListener('click', runExtraction);
  sec.querySelector('#sem-query-btn')?.addEventListener('click', () => {
    const pos = document.getElementById('sem-query-pos')?.value.trim();
    const neg = document.getElementById('sem-query-neg')?.value.trim();
    if (pos) runQueryFrame(currentFrameIdx, pos, neg);
  });
  return [sec];
}

async function checkQueryable(extractor, sec) {
  const data = await fetch(`/api/semantics/queryable`).then(r => r.json()).catch(() => ({}));
  isQueryable = data.queryable || false;
  const qs = (sec || document).querySelector('#sem-query-section');
  if (qs) qs.style.display = isQueryable ? 'flex' : 'none';
}

// ── Main panel setup ──────────────────────────────────────────────
function setupMainPanel() {
  const panel = document.getElementById('tab-semantics');
  if (!panel || panel.dataset.initialized) return;
  panel.dataset.initialized = '1';
  panel.style.cssText = 'display:none;flex-direction:column;flex:1;overflow:hidden;';

  panel.innerHTML = `
    <div id="sem-panels" style="display:flex;flex:1;overflow:hidden;gap:1px">
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0d0d0d;padding:8px">
        <div style="font-size:10px;color:#555;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">Frame</div>
        <img id="sem-img-frame" style="max-width:100%;max-height:calc(100% - 24px);object-fit:contain" src="">
      </div>
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0d0d0d;padding:8px">
        <div style="font-size:10px;color:#555;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">PCA Features</div>
        <img id="sem-img-pca" style="max-width:100%;max-height:calc(100% - 24px);object-fit:contain" src="">
        <div id="sem-pca-status" style="font-size:10px;color:#555;margin-top:4px"></div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0d0d0d;padding:8px">
        <div style="font-size:10px;color:#555;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">Query Similarity</div>
        <img id="sem-img-query" style="max-width:100%;max-height:calc(100% - 24px);object-fit:contain" src="">
        <div id="sem-query-status" style="font-size:10px;color:#555;margin-top:4px">Run a query to see similarity</div>
      </div>
    </div>
    <div id="sem-strip" style="height:88px;background:#0d0d0d;border-top:1px solid #1e1e2e;display:flex;gap:3px;padding:4px 8px;overflow-x:auto;flex-shrink:0;align-items:center"></div>
  `;
}

// ── Load frame visualization ──────────────────────────────────────
async function loadFrame(idx) {
  if (!state.outputDir) return;
  currentFrameIdx = idx;

  const pcaStatus = document.getElementById('sem-pca-status');
  if (pcaStatus) { pcaStatus.textContent = 'Computing…'; }

  const data = await fetch(`/api/semantics/frame_viz?idx=${idx}`).then(r => r.json()).catch(() => ({}));
  if (!data.ok) {
    if (pcaStatus) pcaStatus.textContent = data.error?.split('\n')[0] || 'Error';
    return;
  }

  totalFrames = data.n_frames;
  const frameImg = document.getElementById('sem-img-frame');
  const pcaImg = document.getElementById('sem-img-pca');
  if (frameImg) frameImg.src = data.frame_url;
  if (pcaImg) pcaImg.src = `data:image/png;base64,${data.pca_b64}`;
  if (pcaStatus) pcaStatus.textContent = '';

  // Highlight active frame in strip
  document.querySelectorAll('.sem-thumb').forEach((t, i) => {
    t.style.borderColor = i === idx ? '#2596be' : '#222';
  });
}

async function runQueryFrame(idx, pos, neg) {
  neg = neg || '';
  const qStatus = document.getElementById('sem-query-status');
  if (qStatus) { qStatus.textContent = 'Querying…'; }
  const url = `/api/semantics/query_frame?idx=${idx}&text=${encodeURIComponent(pos)}&neg=${encodeURIComponent(neg)}`;
  const data = await fetch(url).then(r => r.json()).catch(() => ({}));
  if (!data.ok) {
    if (qStatus) qStatus.textContent = data.error?.split('\n')[0] || 'Query failed';
    return;
  }
  const qImg = document.getElementById('sem-img-query');
  if (qImg) qImg.src = `data:image/png;base64,${data.heatmap_b64}`;
  if (qStatus) qStatus.textContent = neg ? `"${pos}" − "${neg}"` : `"${pos}"`;
}

// ── Build frame strip ─────────────────────────────────────────────
async function buildFrameStrip() {
  if (!state.outputDir) return;
  const info = await fetch('/api/preprocess/info').then(r => r.json()).catch(() => ({}));
  if (!info.ok || !info.frames_extracted) return;

  totalFrames = info.frames_extracted;
  const strip = document.getElementById('sem-strip');
  if (!strip) return;
  strip.innerHTML = '';

  const relDir = info.frames_dir.replace('/workspace/outputs/', '');
  // IntersectionObserver for lazy loading
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting && e.target.dataset.src) {
        e.target.src = e.target.dataset.src;
        delete e.target.dataset.src;
        obs.unobserve(e.target);
      }
    });
  }, { root: strip, rootMargin: '0px 200px' });

  for (let i = 0; i < totalFrames; i++) {
    const img = document.createElement('img');
    img.className = 'sem-thumb';
    img.dataset.src = `/outputs/${relDir}/frame_${String(i).padStart(6,'0')}.jpg`;
    img.style.cssText = 'height:72px;width:54px;object-fit:cover;border:2px solid #222;border-radius:2px;cursor:pointer;flex-shrink:0;background:#1a1a1a';
    img.addEventListener('click', () => loadFrame(i));
    strip.appendChild(img);
    obs.observe(img);
  }

  // Load first frame visualization
  loadFrame(0);
}

// ── Extraction SSE ────────────────────────────────────────────────
function runExtraction() {
  const btn = document.getElementById('sem-run-btn');
  const status = document.getElementById('sem-status');
  if (!state.outputDir) { alert('Load a session first'); return; }
  if (btn) btn.disabled = true;
  if (status) { status.textContent = 'Running…'; status.className = 'status-info'; }

  const extractor = state.extractor || 'dinov2';
  const es = new EventSource(`/api/semantics/run?extractor=${encodeURIComponent(extractor)}`);
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    if (status) { status.textContent = ev.msg?.split('\n')[0]; status.className = ev.type === 'done' ? 'status-ok' : ev.type === 'error' ? 'status-err' : 'status-info'; }
    if (ev.type === 'done' || ev.type === 'error') { if (btn) btn.disabled = false; es.close(); }
  };
  es.onerror = () => { if (btn) btn.disabled = false; es.close(); };
}

// ── Activate ──────────────────────────────────────────────────────
async function onActivate() {
  setupMainPanel();
  const data = await fetch('/api/semantics/status').then(r => r.json()).catch(() => ({}));
  if (data.ok && data.cached?.length) {
    const status = document.getElementById('sem-status');
    if (status) { status.textContent = `✓ Cached: ${data.cached.join(', ')}`; status.className = 'status-ok'; }
    const sel = document.getElementById('sem-extractor');
    if (sel && data.cached.length > 0) {
      const match = [...sel.options].find(o => o.value.replace(' ✓','') === data.cached[0]);
      if (match) { match.selected = true; state.extractor = data.cached[0]; }
    }
  }
  // Check if queryable
  await checkQueryable(state.extractor, null);
  buildFrameStrip();
}

registerTab('semantics', { renderSidebar, onActivate });
