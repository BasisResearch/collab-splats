// Central client-side state + tab switching
export const state = {
  outputDir: null,
  videoPath: null,
  creator: 'vggtx',
  conf: 35.0,
  extractor: 'dinov2',
  localizeMethod: null,
  localizeExtractor: 'DISK+LightGlue',
};

const TAB_MODULES = {};

export function registerTab(name, module) {
  TAB_MODULES[name] = module;
}

export function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = '';
  sidebar.appendChild(renderSessionSection());
  if (TAB_MODULES[name]?.renderSidebar) {
    const sections = TAB_MODULES[name].renderSidebar();
    sections.forEach(s => sidebar.appendChild(s));
  }
  if (TAB_MODULES[name]?.onActivate) TAB_MODULES[name].onActivate();
}

function renderSessionSection() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.id = 'section-session';
  sec.innerHTML = `
    <h4>Session</h4>
    <select id="session-select"><option value="">— loading scenes… —</option></select>
    <button class="primary" id="load-session-btn">Load session</button>
    <div id="session-status" class="status-info"></div>
  `;
  sec.querySelector('#load-session-btn').addEventListener('click', loadSession);
  // Populate dropdown from /api/session/list
  fetch('/api/session/list').then(r => r.json()).then(data => {
    const sel = sec.querySelector('#session-select');
    sel.innerHTML = '<option value="">— select scene —</option>';
    (data.sessions || []).forEach(name => {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name;
      if (state.outputDir && state.outputDir.endsWith('/' + name)) o.selected = true;
      sel.appendChild(o);
    });
  }).catch(() => {
    const sel = sec.querySelector('#session-select');
    sel.innerHTML = '<option value="">— error loading scenes —</option>';
  });
  return sec;
}

async function loadSession() {
  const sel = document.getElementById('session-select');
  const name = sel?.value;
  if (!name) return;
  const dir = `/workspace/outputs/${name}`;
  const resp = await fetch('/api/session/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ output_dir: dir }),
  });
  const data = await resp.json();
  if (data.ok) {
    state.outputDir = data.output_dir;
    state.videoPath = data.video_path;
    document.getElementById('session-label').textContent = data.name;
    document.getElementById('session-status').textContent = '✓ ' + data.name;
    document.getElementById('session-status').className = 'status-ok';
    const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
    if (activeTab) switchTab(activeTab);
  } else {
    document.getElementById('session-status').textContent = data.error;
    document.getElementById('session-status').className = 'status-err';
  }
}

export function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('status-msg').textContent = msg || '';
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Initial render — defer so any tab modules loaded in the same page have registered
document.addEventListener('DOMContentLoaded', () => switchTab('preprocess'));
