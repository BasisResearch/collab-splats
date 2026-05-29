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

// Persistent session section — created once, never destroyed on tab switch
function buildSessionSection() {
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
  // Populate dropdown once
  fetch('/api/session/list').then(r => r.json()).then(data => {
    const sel = document.getElementById('session-select');
    if (!sel) return;
    sel.innerHTML = '<option value="">— select scene —</option>';
    (data.sessions || []).forEach(name => {
      const o = document.createElement('option');
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    });
  }).catch(() => {});
  return sec;
}

// Divider between session and tab-specific sections
function buildTabSections() {
  const div = document.createElement('div');
  div.id = 'sidebar-tab-sections';
  div.style.cssText = 'display:flex;flex-direction:column;gap:16px';
  return div;
}

export function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  // Only replace tab-specific sections; session section stays intact
  const tabSections = document.getElementById('sidebar-tab-sections');
  if (tabSections) {
    tabSections.innerHTML = '';
    if (TAB_MODULES[name]?.renderSidebar) {
      TAB_MODULES[name].renderSidebar().forEach(s => tabSections.appendChild(s));
    }
  }
  if (TAB_MODULES[name]?.onActivate) TAB_MODULES[name].onActivate();
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
  const statusEl = document.getElementById('session-status');
  if (data.ok) {
    state.outputDir = data.output_dir;
    state.videoPath = data.video_path;
    document.getElementById('session-label').textContent = data.name;
    if (statusEl) { statusEl.textContent = '✓ ' + data.name; statusEl.className = 'status-ok'; }
    // Refresh tab-specific sections with new session context
    const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
    if (activeTab) switchTab(activeTab);
  } else {
    if (statusEl) { statusEl.textContent = data.error; statusEl.className = 'status-err'; }
  }
}

export function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('status-msg').textContent = msg || '';
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

document.addEventListener('DOMContentLoaded', () => {
  // Build sidebar structure once: permanent session section + tab-specific slot
  const sidebar = document.getElementById('sidebar');
  sidebar.appendChild(buildSessionSection());
  sidebar.appendChild(buildTabSections());
  switchTab('preprocess');
});
