import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Semantic methods</h4>
    <label>Extractor</label>
    <select id="sem-extractor"><option value="">— loading… —</option></select>
    <button class="primary" id="sem-run-btn" style="margin-top:8px">&#9654; Extract features</button>
    <div id="sem-status" class="status-info"></div>
  `;
  // Load methods + cached status in parallel; prefer cached extractors
  Promise.all([
    fetch('/api/semantics/methods').then(r => r.json()).catch(() => ({methods: ['dinov2']})),
    fetch('/api/semantics/status').then(r => r.json()).catch(() => ({cached: []})),
  ]).then(([methodsData, statusData]) => {
    const sel = sec.querySelector('#sem-extractor');
    const cached = new Set(statusData.cached || []);
    const methods = methodsData.methods || ['dinov2'];
    sel.innerHTML = '';
    // Cached extractors first, then uncached
    const ordered = [...methods.filter(m => cached.has(m)), ...methods.filter(m => !cached.has(m))];
    ordered.forEach(m => {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = cached.has(m) ? `${m} ✓` : m;
      if (m === state.extractor || (cached.size > 0 && cached.has(m) && !state.extractor)) o.selected = true;
      sel.appendChild(o);
    });
    // Show cache status
    const status = sec.querySelector('#sem-status');
    if (status && cached.size > 0) {
      status.textContent = `✓ Cached: ${[...cached].join(', ')}`;
      status.className = 'status-ok';
    }
    // Auto-select first cached extractor
    if (cached.size > 0) {
      const first = ordered[0];
      sel.value = first;
      state.extractor = first;
      fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({extractor: first}) });
    }
  });
  sec.querySelector('#sem-extractor').addEventListener('change', e => {
    state.extractor = e.target.value.replace(' ✓', '');
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({extractor: state.extractor}) });
  });
  sec.querySelector('#sem-run-btn').addEventListener('click', runSemantics);
  return [sec];
}

function runSemantics() {
  const log = document.getElementById('log-semantics');
  const btn = document.getElementById('sem-run-btn');
  if (!state.outputDir) { alert('Load a session first'); return; }
  if (log) log.textContent = '';
  if (btn) btn.disabled = true;
  const status = document.getElementById('sem-status');
  if (status) { status.textContent = 'Running…'; status.className = 'status-info'; }

  const extractor = state.extractor || document.getElementById('sem-extractor')?.value.replace(' ✓', '') || 'dinov2';
  const es = new EventSource(`/api/semantics/run?extractor=${encodeURIComponent(extractor)}`);
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    if (log) {
      const line = document.createElement('div');
      line.textContent = ev.msg;
      if (ev.type === 'done') line.className = 'ok';
      if (ev.type === 'error') line.className = 'err';
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    }
    if (ev.type === 'done') {
      if (status) { status.textContent = '✓ ' + ev.msg; status.className = 'status-ok'; }
      if (btn) btn.disabled = false; es.close();
    }
    if (ev.type === 'error') {
      if (status) { status.textContent = ev.msg.split('\n')[0]; status.className = 'status-err'; }
      if (btn) btn.disabled = false; es.close();
    }
  };
  es.onerror = () => { if (btn) btn.disabled = false; es.close(); };
}

async function onActivate() {
  const data = await fetch('/api/semantics/status').then(r => r.json()).catch(() => ({}));
  if (!data.ok || !data.cached?.length) return;
  const status = document.getElementById('sem-status');
  if (status) { status.textContent = `✓ Cached: ${data.cached.join(', ')}`; status.className = 'status-ok'; }
  // Pre-select first cached extractor
  const sel = document.getElementById('sem-extractor');
  if (sel && data.cached.length > 0) {
    const match = [...sel.options].find(o => o.value.replace(' ✓','') === data.cached[0]);
    if (match) { match.selected = true; state.extractor = data.cached[0]; }
  }
}

registerTab('semantics', { renderSidebar, onActivate });
