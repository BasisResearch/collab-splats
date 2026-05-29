import { state, registerTab, setProgress } from './state.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

let renderer, scene, camera, controls;
let currentPoints = null;
let currentMesh = null;
let currentFrustums = null;
let lastGroundPlane = null;
// sceneRoot: THREE.Group with ground plane transform applied once.
// Both PLY and mesh are added to this group WITHOUT per-geometry GP transforms.
// This guarantees identical orientation when switching between them.
let sceneRoot = null;
// Normalization in raw world space (shared by PLY, mesh, frustums)
let normCenter = null;
let normScale = 1;
let plyNormCenter = null;
let plyNormScale = 1;

function initThree() {
  const canvas = document.getElementById('three-canvas');
  if (!canvas || renderer) return;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0a0a0f);

  scene = new THREE.Scene();
  // Ambient + directional lights for mesh shading
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 2, 3);
  scene.add(dir);

  camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.0001, 1000);
  camera.position.set(0, 0, 2.5);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  // Prevent gimbal lock at poles: allow full range except the singularity points
  controls.minPolarAngle = 0.05;   // ~3° from straight-down view
  controls.maxPolarAngle = Math.PI - 0.05;  // ~3° from straight-up view

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  (function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();
}

// Build or rebuild the sceneRoot Group with the ground plane transform applied once.
// Both PLY and mesh are children of this group — guarantees identical orientation.
function buildSceneRoot(gp) {
  if (sceneRoot) {
    scene.remove(sceneRoot);
    sceneRoot = null;
  }
  sceneRoot = new THREE.Group();
  if (gp) {
    const { R, t } = gp;
    const m = new THREE.Matrix4().set(
      R[0][0], R[0][1], R[0][2], t[0],
      R[1][0], R[1][1], R[1][2], t[1],
      R[2][0], R[2][1], R[2][2], t[2],
      0, 0, 0, 1
    );
    sceneRoot.applyMatrix4(m);
  }
  scene.add(sceneRoot);
  return sceneRoot;
}

// Normalize geometry: center + scale. First call computes norm params; later calls reuse them.
function normalizeGeo(geo, fresh) {
  if (fresh) { normCenter = null; normScale = 1; }
  geo.computeBoundingBox();
  if (!normCenter) {
    normCenter = new THREE.Vector3();
    geo.boundingBox.getCenter(normCenter);
    const size = new THREE.Vector3();
    geo.boundingBox.getSize(size);
    normScale = 2 / Math.max(size.x, size.y, size.z, 0.001);
  }
  geo.translate(-normCenter.x, -normCenter.y, -normCenter.z);
  geo.scale(normScale, normScale, normScale);
}

function loadPLY(url, gp, pointSize) {
  pointSize = pointSize || 0.004;
  if (!renderer) initThree();
  originalColors = null;
  lastSimilarityColors = null;
  // (Re)build scene root with ground plane transform — both PLY and mesh share this group
  buildSceneRoot(gp);
  setProgress(10, 'Loading pointcloud…');
  new PLYLoader().load(url, geo => {
    if (currentPoints) { sceneRoot.remove(currentPoints); currentPoints.geometry.dispose(); currentPoints = null; }
    // Normalize in raw world space (no per-geometry GP transform)
    normalizeGeo(geo, true);
    plyNormCenter = normCenter ? normCenter.clone() : null;
    plyNormScale = normScale;
    const mat = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: !!geo.attributes.color,
      sizeAttenuation: true,
    });
    if (!geo.attributes.color) mat.color.set(0x2596be);
    currentPoints = new THREE.Points(geo, mat);
    sceneRoot.add(currentPoints);
    setProgress(100, `${geo.attributes.position.count.toLocaleString()} points`);
  }, xhr => {
    if (xhr.total) setProgress(Math.round(xhr.loaded / xhr.total * 90), 'Loading…');
  });
}

function loadMesh(url, gp) {
  if (!renderer) initThree();
  // Ensure sceneRoot exists (may be loading mesh before PLY on Mesh-first flow)
  if (!sceneRoot) buildSceneRoot(gp);
  setProgress(10, 'Loading mesh…');
  new PLYLoader().load(url, geo => {
    if (currentMesh) { sceneRoot.remove(currentMesh); currentMesh.geometry.dispose(); currentMesh = null; }
    // Same normalization as PLY — no per-geometry GP transform
    normalizeGeo(geo, false);
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhongMaterial({
      vertexColors: !!geo.attributes.color,
      side: THREE.DoubleSide,
      shininess: 30,
    });
    if (!geo.attributes.color) mat.color.set(0x888888);
    currentMesh = new THREE.Mesh(geo, mat);
    sceneRoot.add(currentMesh);
    setProgress(100, 'Mesh loaded');
  }, xhr => {
    if (xhr.total) setProgress(Math.round(xhr.loaded / xhr.total * 90), 'Loading mesh…');
  });
}

// Store original colors + last similarity result separately
let originalColors = null;
let lastSimilarityColors = null;  // preserved across toggle off/on

function resetPointColors() {
  if (!currentPoints || !originalColors) return;
  const geo = currentPoints.geometry;
  if (geo.attributes.color) {
    geo.attributes.color.array.set(originalColors);
    geo.attributes.color.needsUpdate = true;
  }
}

function restoreLastSimilarity() {
  if (!currentPoints || !lastSimilarityColors) return false;
  const geo = currentPoints.geometry;
  if (geo.attributes.color) {
    geo.attributes.color.array.set(lastSimilarityColors);
    geo.attributes.color.needsUpdate = true;
    return true;
  }
  return false;
}

async function runSimilarityQuery(pos, neg, extractor, statusEl) {
  if (!currentPoints) {
    if (statusEl) { statusEl.textContent = 'Load scene first'; statusEl.className = 'status-err'; }
    return;
  }
  const geo = currentPoints.geometry;
  if (!geo.attributes.color) {
    if (statusEl) { statusEl.textContent = 'Pointcloud has no color attribute'; statusEl.className = 'status-err'; }
    return;
  }

  // Save original colors on first query
  if (!originalColors) {
    originalColors = new Float32Array(geo.attributes.color.array);
  }

  const url = `/api/visualize/similarity?pos=${encodeURIComponent(pos)}&neg=${encodeURIComponent(neg)}&extractor=${extractor}`;
  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (!data.ok) {
      if (statusEl) { statusEl.textContent = data.error?.split('\n')[0] || 'Query failed'; statusEl.className = 'status-err'; }
      return;
    }
    // Decode base64 → Uint8Array (N*3 bytes) → update Three.js color attribute
    const raw = atob(data.colors_b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const colors = geo.attributes.color.array;
    const n = Math.min(data.n_points, colors.length / 3);
    for (let i = 0; i < n; i++) {
      colors[i * 3]     = bytes[i * 3]     / 255;
      colors[i * 3 + 1] = bytes[i * 3 + 1] / 255;
      colors[i * 3 + 2] = bytes[i * 3 + 2] / 255;
    }
    // Save similarity colors so toggling off/on restores them without re-querying
    lastSimilarityColors = new Float32Array(geo.attributes.color.array);
    geo.attributes.color.needsUpdate = true;
    if (statusEl) { statusEl.textContent = `✓ "${pos}"${neg ? ' − "' + neg + '"' : ''}`; statusEl.className = 'status-ok'; }
  } catch (e) {
    if (statusEl) { statusEl.textContent = String(e); statusEl.className = 'status-err'; }
  }
}

function _applyGP(p, gp) {
  if (!gp) return p;
  const { R, t } = gp;
  return [
    R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2] + t[0],
    R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2] + t[1],
    R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2] + t[2],
  ];
}

function _normPt(p) {
  // Always use PLY normalization for frustum alignment
  const c = plyNormCenter, s = plyNormScale;
  return [
    (p[0] - (c?.x || 0)) * s,
    (p[1] - (c?.y || 0)) * s,
    (p[2] - (c?.z || 0)) * s,
  ];
}

async function loadFrustums(gp) {
  if (currentFrustums) { scene.remove(currentFrustums); currentFrustums.geometry.dispose(); currentFrustums = null; }
  const resp = await fetch('/api/visualize/frustums');
  const data = await resp.json();
  if (!data.ok || !data.frustums?.length) { setProgress(0, data.error || 'No cameras'); return; }

  const verts = [];

  // Frustums are in raw world space — normalize same as PLY, sceneRoot applies GP
  data.frustums.forEach(f => {
    const c = _normPt(f.center);
    const corners = f.corners.map(corner => _normPt(corner));

    corners.forEach(co => { verts.push(...c, ...co); });
    [0,1,2,3].forEach(i => { verts.push(...corners[i], ...corners[(i+1) % 4]); });
  });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x2596be, opacity: 0.6, transparent: true });
  currentFrustums = new THREE.LineSegments(geo, mat);
  if (!sceneRoot) buildSceneRoot(gp);
  sceneRoot.add(currentFrustums);
  setProgress(100, `${data.n_cameras} cameras`);
}

async function loadScene() {
  if (!state.outputDir) return;
  const resp = await fetch('/api/visualize/status');
  const data = await resp.json();
  if (!data.ok) { setProgress(0, data.error || 'No session'); return; }

  // Sync creator dropdown
  if (data.creator) {
    state.creator = data.creator;
    const sel = document.getElementById('viz-creator');
    if (sel && data.available_backends?.length) {
      sel.innerHTML = '';
      data.available_backends.forEach(b => {
        const o = document.createElement('option');
        o.value = b; o.textContent = b;
        if (b === data.creator) o.selected = true;
        sel.appendChild(o);
      });
    }
  }

  const gp = data.ground_plane || null;
  lastGroundPlane = gp;
  if (data.ply_url) {
    loadPLY(data.ply_url, gp);
  } else {
    setProgress(0, 'No pointcloud found for this scene');
  }
}

function renderSidebar() {
  const pc = document.createElement('div');
  pc.className = 'sidebar-section';
  pc.innerHTML = `
    <h4>Pointcloud</h4>
    <label>Creator</label>
    <select id="viz-creator"><option value="">— loading… —</option></select>
    <button class="primary" id="viz-load-btn" style="margin-top:4px">Load scene</button>
  `;
  const sem = document.createElement('div');
  sem.className = 'sidebar-section';
  sem.innerHTML = `
    <h4>Semantic methods</h4>
    <select id="viz-extractor"><option value="dinov2">DINOv2</option></select>
    <label style="display:flex;align-items:center;gap:6px;margin-top:4px">
      <input type="checkbox" id="viz-sem-toggle"> Show similarity
    </label>
    <div id="viz-sem-inputs" style="display:none;flex-direction:column;gap:4px;margin-top:4px">
      <input type="text" id="viz-pos-query" placeholder="positive query" value="tree">
      <input type="text" id="viz-neg-query" placeholder="negative query" value="background, ground, sky">
      <button class="primary" id="viz-query-btn">&#9654; Query</button>
      <button id="viz-reset-colors-btn" style="background:none;border:1px solid #555;color:#aaa;padding:5px;border-radius:3px;font-family:monospace;font-size:11px;cursor:pointer;margin-top:2px">Reset to original colors</button>
      <div id="viz-sem-status" class="status-info"></div>
    </div>
  `;
  // Populate semantic extractor from registry
  fetch('/api/semantics/methods').then(r => r.json()).then(d => {
    const sel = sem.querySelector('#viz-extractor');
    sel.innerHTML = '';
    (d.methods || []).forEach(m => {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      if (m === state.extractor) o.selected = true;
      sel.appendChild(o);
    });
  }).catch(() => {});

  sem.querySelector('#viz-sem-toggle').addEventListener('change', e => {
    sem.querySelector('#viz-sem-inputs').style.display = e.target.checked ? 'flex' : 'none';
    if (e.target.checked) {
      // Restore last similarity result if available, otherwise wait for user to query
      restoreLastSimilarity();
    } else {
      // Save current similarity state then revert to original colors
      if (currentPoints?.geometry?.attributes?.color) {
        lastSimilarityColors = new Float32Array(currentPoints.geometry.attributes.color.array);
      }
      resetPointColors();
    }
  });
  sem.querySelector('#viz-query-btn').addEventListener('click', () => {
    const pos = sem.querySelector('#viz-pos-query').value.trim();
    const neg = sem.querySelector('#viz-neg-query').value.trim();
    const ext = sem.querySelector('#viz-extractor').value;
    const status = sem.querySelector('#viz-sem-status');
    if (!pos) { if (status) { status.textContent = 'Enter a positive query'; status.className = 'status-err'; } return; }
    if (status) { status.textContent = 'Computing…'; status.className = 'status-info'; }
    runSimilarityQuery(pos, neg, ext, status);
  });
  sem.querySelector('#viz-reset-colors-btn').addEventListener('click', resetPointColors);

  const view = document.createElement('div');
  view.className = 'sidebar-section';
  view.innerHTML = `
    <h4>View</h4>
    <div style="display:flex;gap:6px;margin-bottom:6px">
      <button id="viz-btn-pc" style="flex:1;padding:6px 0;border-radius:3px;border:1px solid #2596be;background:#2596be;color:#000;font-family:monospace;font-size:11px;font-weight:700;cursor:pointer">☁ Cloud</button>
      <button id="viz-btn-mesh" style="flex:1;padding:6px 0;border-radius:3px;border:1px solid #444;background:none;color:#888;font-family:monospace;font-size:11px;cursor:pointer">⬛ Mesh</button>
    </div>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-frustums"> Show cameras</label>
    <div style="margin-top:8px">
      <button id="viz-detect-gp-btn" style="width:100%;padding:6px;border-radius:3px;border:1px solid #555;background:none;color:#aaa;font-family:monospace;font-size:11px;cursor:pointer">⟳ Detect ground plane</button>
      <div id="viz-gp-status" class="status-info" style="margin-top:4px"></div>
    </div>
    <div style="margin-top:10px;display:flex;flex-direction:column;gap:4px;display:none" id="viz-mesh-gen-section">
      <h4 style="font-size:10px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:1px">Mesh generation</h4>
      <label>voxel_size: <input type="number" id="viz-voxel" value="0.005" step="0.001" min="0.001" style="width:80px;padding:2px 4px"></label>
      <label>sdf_trunc: <input type="number" id="viz-sdf" value="0.02" step="0.005" min="0.001" style="width:80px;padding:2px 4px"></label>
      <label>depth_trunc: <input type="number" id="viz-depth" value="1.0" step="0.5" min="0.1" style="width:80px;padding:2px 4px"></label>
      <button id="viz-run-mesh-btn" style="width:100%;padding:6px;border-radius:3px;border:1px solid #555;background:none;color:#aaa;font-family:monospace;font-size:11px;cursor:pointer">⚙ Run mesh</button>
      <div id="viz-mesh-gen-status" class="status-info"></div>
    </div>
  `;

  pc.querySelector('#viz-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });
  pc.querySelector('#viz-load-btn').addEventListener('click', loadScene);

  function setActiveMode(mode) {
    const btnPc = document.getElementById('viz-btn-pc');
    const btnMesh = document.getElementById('viz-btn-mesh');
    if (!btnPc || !btnMesh) return;
    if (mode === 'cloud') {
      btnPc.style.cssText = 'flex:1;padding:6px 0;border-radius:3px;border:1px solid #2596be;background:#2596be;color:#000;font-family:monospace;font-size:11px;font-weight:700;cursor:pointer';
      btnMesh.style.cssText = 'flex:1;padding:6px 0;border-radius:3px;border:1px solid #444;background:none;color:#888;font-family:monospace;font-size:11px;cursor:pointer';
      if (currentPoints) currentPoints.visible = true;
      if (currentMesh) { sceneRoot?.remove(currentMesh); currentMesh.geometry.dispose(); currentMesh = null; }
      const meshGenSec = document.getElementById('viz-mesh-gen-section');
      if (meshGenSec) meshGenSec.style.display = 'none';
    } else {
      btnMesh.style.cssText = 'flex:1;padding:6px 0;border-radius:3px;border:1px solid #2596be;background:#2596be;color:#000;font-family:monospace;font-size:11px;font-weight:700;cursor:pointer';
      btnPc.style.cssText = 'flex:1;padding:6px 0;border-radius:3px;border:1px solid #444;background:none;color:#888;font-family:monospace;font-size:11px;cursor:pointer';
      if (currentPoints) currentPoints.visible = false;
      // Show mesh generation options when mesh mode is active
      const meshGenSec = document.getElementById('viz-mesh-gen-section');
      if (meshGenSec) meshGenSec.style.display = 'flex';
    }
  }

  view.querySelector('#viz-btn-pc').addEventListener('click', () => setActiveMode('cloud'));

  view.querySelector('#viz-btn-mesh').addEventListener('click', async () => {
    const resp = await fetch('/api/visualize/status');
    const data = await resp.json();
    if (data.mesh_url) {
      loadMesh(data.mesh_url, data.ground_plane);
      setActiveMode('mesh');
    } else {
      setProgress(0, 'No mesh found for this scene');
    }
  });

  // Camera frustums toggle
  view.querySelector('#viz-frustums').addEventListener('change', e => {
    if (e.target.checked) {
      loadFrustums(lastGroundPlane);
    } else {
      if (currentFrustums) { sceneRoot?.remove(currentFrustums); currentFrustums.geometry.dispose(); currentFrustums = null; }
    }
  });

  // Mesh generation
  view.querySelector('#viz-run-mesh-btn').addEventListener('click', () => {
    const voxel = view.querySelector('#viz-voxel').value || 0.005;
    const sdf = view.querySelector('#viz-sdf').value || 0.02;
    const depth = view.querySelector('#viz-depth').value || 1.0;
    const btn = view.querySelector('#viz-run-mesh-btn');
    const status = view.querySelector('#viz-mesh-gen-status');
    btn.disabled = true;
    if (status) { status.textContent = 'Running…'; status.className = 'status-info'; }
    const es = new EventSource(`/api/visualize/run_mesh?voxel_size=${voxel}&sdf_trunc=${sdf}&depth_trunc=${depth}`);
    es.onmessage = e => {
      const ev = JSON.parse(e.data);
      if (status) { status.textContent = ev.msg?.split('\n')[0]; status.className = ev.type === 'done' ? 'status-ok' : ev.type === 'error' ? 'status-err' : 'status-info'; }
      if (ev.type === 'done' || ev.type === 'error') { btn.disabled = false; es.close(); }
    };
    es.onerror = () => { btn.disabled = false; es.close(); };
  });

  // Ground plane detection
  view.querySelector('#viz-detect-gp-btn').addEventListener('click', async () => {
    const btn = view.querySelector('#viz-detect-gp-btn');
    const status = view.querySelector('#viz-gp-status');
    btn.disabled = true;
    if (status) { status.textContent = 'Detecting…'; status.className = 'status-info'; }
    const data = await fetch('/api/visualize/detect_ground_plane').then(r => r.json()).catch(e => ({ok: false, error: String(e)}));
    btn.disabled = false;
    if (data.ok) {
      if (status) { status.textContent = '✓ Saved — reloading scene…'; status.className = 'status-ok'; }
      lastGroundPlane = data.ground_plane;
      normCenter = null;  // Force re-normalization on reload
      loadScene();  // Reload with new ground plane
    } else {
      if (status) { status.textContent = data.error?.split('\n')[0] || 'Failed'; status.className = 'status-err'; }
    }
  });

  return [pc, sem, view];
}

registerTab('visualize', {
  renderSidebar,
  onActivate() { initThree(); loadScene(); },
});
