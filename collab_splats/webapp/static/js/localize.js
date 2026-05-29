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
    <button class="primary" id="loc-run-btn" style="margin-top:8px">&#9654; Localize</button>
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

registerTab('localize', { renderSidebar });
