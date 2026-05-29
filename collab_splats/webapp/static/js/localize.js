import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const pc = document.createElement('div');
  pc.className = 'sidebar-section';
  pc.innerHTML = `
    <h4>Pointcloud</h4>
    <select id="loc-method"><option value="">— select method —</option></select>
  `;

  const loc = document.createElement('div');
  loc.className = 'sidebar-section';
  loc.innerHTML = `
    <h4>Localize method</h4>
    <label>Extractor</label>
    <select id="loc-extractor">
      <option value="DISK+LightGlue">DISK+LightGlue</option>
      <option value="XFeat+MNN">XFeat+MNN</option>
    </select>
    <label style="margin-top:8px">Query image path</label>
    <input type="text" id="loc-query" placeholder="/path/to/query.jpg">
    <button id="loc-sample-btn" style="width:100%;padding:5px;margin-top:4px;border-radius:3px;border:1px solid #555;background:none;color:#aaa;font-family:monospace;font-size:11px;cursor:pointer">Use out-of-sample frame</button>
    <button class="primary" id="loc-run-btn" style="margin-top:6px">&#9654; Localize</button>
    <div id="loc-status" class="status-info"></div>
  `;

  // Populate method dropdown
  fetch('/api/localize/methods').then(r => r.json()).then(data => {
    const sel = pc.querySelector('#loc-method');
    (data.methods || []).forEach(m => {
      const o = document.createElement('option'); o.value = m; o.textContent = m; sel.appendChild(o);
    });
    if (state.creator) sel.value = state.creator;
  }).catch(() => {});

  pc.querySelector('#loc-method').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value, localize_method: e.target.value}) });
  });

  loc.querySelector('#loc-extractor').addEventListener('change', e => {
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({localize_extractor: e.target.value}) });
  });

  loc.querySelector('#loc-run-btn').addEventListener('click', () => {
    const query = document.getElementById('loc-query')?.value.trim() || '';
    runLocalize(query);
  });

  // Load a sample out-of-sample image path on click
  loc.querySelector('#loc-sample-btn').addEventListener('click', async () => {
    const btn = loc.querySelector('#loc-sample-btn');
    btn.textContent = 'Loading…';
    const data = await fetch('/api/localize/sample_query').then(r => r.json()).catch(() => ({}));
    if (data.ok) {
      const inp = document.getElementById('loc-query');
      if (inp) inp.value = data.path;
      btn.textContent = `Sample: ${data.scene}`;
    } else {
      btn.textContent = 'No other scenes found';
    }
  });

  // Auto-load sample query on activate
  fetch('/api/localize/sample_query').then(r => r.json()).then(data => {
    if (data.ok) {
      const inp = document.getElementById('loc-query');
      if (inp && !inp.value) inp.value = data.path;
      const btn = document.getElementById('loc-sample-btn');
      if (btn) btn.textContent = `Sample: ${data.scene}`;
    }
  }).catch(() => {});

  return [pc, loc];
}

function runLocalize(queryPath) {
  const log = document.getElementById('log-localize');
  const btn = document.getElementById('loc-run-btn');
  const status = document.getElementById('loc-status');
  if (log) log.textContent = '';
  if (btn) btn.disabled = true;
  if (status) { status.textContent = 'Running…'; status.className = 'status-info'; }

  const es = new EventSource(`/api/localize/run?query_path=${encodeURIComponent(queryPath)}`);
  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    if (log) {
      const line = document.createElement('div');
      if (ev.type === 'done' && ev.results) {
        line.innerHTML = `<span class="ok">${ev.msg}</span><br><pre>${JSON.stringify(ev.results, null, 2)}</pre>`;
      } else {
        line.textContent = ev.msg;
        if (ev.type === 'error') line.className = 'err';
      }
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
  // Refresh method dropdown with latest available backends on every tab switch
  const data = await fetch('/api/localize/methods').then(r => r.json()).catch(() => ({}));
  const sel = document.getElementById('loc-method');
  if (!sel || !data.methods?.length) return;
  const current = sel.value;
  sel.innerHTML = '<option value="">— select method —</option>';
  data.methods.forEach(m => {
    const o = document.createElement('option'); o.value = m; o.textContent = m;
    if (m === current || m === state.creator) o.selected = true;
    sel.appendChild(o);
  });
  // Auto-select first method and update session
  if (!current && data.methods.length > 0) {
    sel.value = data.methods[0];
    state.creator = data.methods[0];
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: data.methods[0], localize_method: data.methods[0]}) });
  }
  const status = document.getElementById('loc-status');
  if (status && data.methods.length > 0) {
    status.textContent = `${data.methods.length} method(s) available`;
    status.className = 'status-info';
  }
}

registerTab('localize', { renderSidebar, onActivate });
