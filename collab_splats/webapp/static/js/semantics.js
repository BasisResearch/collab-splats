import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Semantic methods</h4>
    <label>Extractor</label>
    <select id="sem-extractor">
      <option value="dinov2">DINOv2</option>
      <option value="sam">SAM</option>
    </select>
    <button class="primary" id="sem-run-btn" style="margin-top:8px">&#9654; Extract features</button>
    <div id="sem-status" class="status-info"></div>
  `;
  sec.querySelector('#sem-extractor').addEventListener('change', e => {
    state.extractor = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({extractor: e.target.value}) });
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

  const es = new EventSource('/api/semantics/run');
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

registerTab('semantics', { renderSidebar });
