import { state, registerTab, setProgress } from './state.js';

function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Pointcloud</h4>
    <label>Creator</label>
    <select id="rc-creator">
      <option value="vggtx">vggtx</option>
      <option value="mapanything">mapanything</option>
      <option value="vggt_omega">vggt_omega</option>
    </select>
    <label>Conf threshold: <span id="rc-conf-val">35</span></label>
    <input type="range" id="rc-conf" min="0" max="100" value="35">
    <div style="display:flex;gap:8px;margin-top:4px">
      <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="rc-ba" disabled> BA</label>
      <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="rc-lc" disabled> LC</label>
    </div>
    <button class="primary" id="rc-run-btn" style="margin-top:8px">&#9654; Run reconstruction</button>
    <div id="rc-status" class="status-info"></div>
  `;
  sec.querySelector('#rc-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ creator: e.target.value }),
    });
  });
  sec.querySelector('#rc-conf').addEventListener('input', e => {
    sec.querySelector('#rc-conf-val').textContent = e.target.value;
    state.conf = parseFloat(e.target.value);
    fetch('/api/session/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conf: parseFloat(e.target.value) }),
    });
  });
  sec.querySelector('#rc-run-btn').addEventListener('click', runReconstruction);
  return [sec];
}

function runReconstruction() {
  const log = document.getElementById('log-reconstruct');
  const btn = document.getElementById('rc-run-btn');
  if (!state.outputDir) { alert('Load a session first'); return; }
  if (log) log.textContent = '';
  if (btn) btn.disabled = true;
  const status = document.getElementById('rc-status');
  if (status) { status.textContent = 'Running…'; status.className = 'status-info'; }

  const es = new EventSource('/api/reconstruct/run');
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
    if (ev.type === 'progress') setProgress(ev.pct || 0, ev.msg);
    if (ev.type === 'done') {
      setProgress(100, 'Done');
      if (status) { status.textContent = '✓ ' + ev.msg; status.className = 'status-ok'; }
      if (btn) btn.disabled = false;
      es.close();
    }
    if (ev.type === 'error') {
      if (status) { status.textContent = ev.msg.split('\n')[0]; status.className = 'status-err'; }
      if (btn) btn.disabled = false;
      es.close();
    }
  };
  es.onerror = () => { if (btn) btn.disabled = false; es.close(); };
}

registerTab('reconstruct', { renderSidebar });
