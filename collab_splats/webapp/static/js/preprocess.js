import { state, registerTab, setProgress } from './state.js';

// ── Sidebar section ──────────────────────────────────────────────
function renderSidebar() {
  const sec = document.createElement('div');
  sec.className = 'sidebar-section';
  sec.innerHTML = `
    <h4>Extraction</h4>
    <label>Method</label>
    <select id="extract-method">
      <option value="optical_flow">Optical flow</option>
      <option value="balanced">Balanced FPS</option>
    </select>
    <label>Max frames: <span id="max-frames-val">200</span></label>
    <input type="range" id="max-frames" min="20" max="500" value="200">
    <label>Min disparity: <span id="min-disp-val">50</span></label>
    <input type="range" id="min-disp" min="5" max="200" value="50">
    <button class="primary" id="extract-btn">&#9654; Extract frames</button>
    <div id="extract-status" class="status-info"></div>
  `;
  sec.querySelector('#max-frames').addEventListener('input', e => {
    sec.querySelector('#max-frames-val').textContent = e.target.value;
  });
  sec.querySelector('#min-disp').addEventListener('input', e => {
    sec.querySelector('#min-disp-val').textContent = e.target.value;
  });
  sec.querySelector('#extract-btn').addEventListener('click', startExtraction);
  return [sec];
}

// ── Load video info ───────────────────────────────────────────────
async function loadVideoInfo() {
  const resp = await fetch('/api/preprocess/info');
  const data = await resp.json();
  if (!data.ok) return;

  const video = document.getElementById('main-video');
  if (video) video.src = '/api/session/video';

  if (data.video) {
    const m = data.video;
    const meta = document.getElementById('video-meta');
    if (meta) meta.innerHTML = `
      <div style="color:#2596be;font-weight:bold;margin-bottom:6px">VIDEO INFO</div>
      <div>${m.width}&#xd7;${m.height} &middot; ${(m.fps||0).toFixed(1)} fps</div>
      <div>${m.total_frames} frames &middot; ${(m.duration_s||0).toFixed(1)}s</div>
    `;
  }

  if (data.frames_extracted > 0) {
    renderFrameStrip(data.frames_extracted);
    const status = document.getElementById('extract-status');
    if (status) {
      status.textContent = `✓ ${data.frames_extracted} frames`;
      status.className = 'status-ok';
    }
  }
}

// ── Frame strip ───────────────────────────────────────────────────
// Use IntersectionObserver to only load frames entering the visible viewport.
// With 200+ frames, eagerly setting src blocks the browser.
const _stripObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting && e.target.dataset.src) {
      e.target.src = e.target.dataset.src;
      delete e.target.dataset.src;
      _stripObserver.unobserve(e.target);
    }
  });
}, { root: document.getElementById('frame-strip'), rootMargin: '0px 200px' });

function renderFrameStrip(count) {
  const strip = document.getElementById('frame-strip');
  if (!strip) return;
  strip.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const img = document.createElement('img');
    const url = `/api/preprocess/frame/${i}`;
    // Don't set src yet — use data-src and let IntersectionObserver trigger load
    img.dataset.src = url;
    img.style.cssText = 'height:72px;width:54px;background:#1a1a1a;border:2px solid #222;border-radius:2px;cursor:pointer;flex-shrink:0;object-fit:cover';
    img.addEventListener('click', () => seekVideo(i, count));
    img.addEventListener('mouseenter', () => { img.style.borderColor = '#2596be'; });
    img.addEventListener('mouseleave', () => { img.style.borderColor = '#222'; });
    strip.appendChild(img);
    _stripObserver.observe(img);
  }
}

function seekVideo(frameIdx, totalFrames) {
  const video = document.getElementById('main-video');
  if (!video || !video.duration) return;
  video.currentTime = (frameIdx / totalFrames) * video.duration;
}

// ── SSE extraction ────────────────────────────────────────────────
function startExtraction() {
  if (!state.outputDir) {
    alert('Load a session first');
    return;
  }
  const method = document.getElementById('extract-method')?.value || 'optical_flow';
  const maxFrames = document.getElementById('max-frames')?.value || 200;
  const minDisp = document.getElementById('min-disp')?.value || 50;

  const btn = document.getElementById('extract-btn');
  if (btn) btn.disabled = true;
  const status = document.getElementById('extract-status');
  if (status) { status.textContent = 'Extracting…'; status.className = 'status-info'; }

  const url = `/api/preprocess/extract?method=${method}&max_frames=${maxFrames}&min_disparity=${minDisp}`;
  const es = new EventSource(url);

  es.onmessage = e => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'progress') {
      setProgress(ev.pct, ev.msg);
    } else if (ev.type === 'done') {
      setProgress(100, ev.msg);
      if (status) { status.textContent = `✓ ${ev.msg}`; status.className = 'status-ok'; }
      if (btn) btn.disabled = false;
      es.close();
      loadVideoInfo();
    } else if (ev.type === 'error') {
      setProgress(0, '');
      if (status) { status.textContent = ev.msg; status.className = 'status-err'; }
      if (btn) btn.disabled = false;
      es.close();
    }
  };
  es.onerror = () => { if (btn) btn.disabled = false; es.close(); };
}

// ── Register + init ────────────────────────────────────────────────
registerTab('preprocess', { renderSidebar, onActivate: loadVideoInfo });
