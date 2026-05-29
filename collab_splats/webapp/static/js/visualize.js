import { state, registerTab, setProgress } from './state.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

let renderer, scene, camera, controls;
let currentPoints = null;
let currentMesh = null;
let currentFrustums = null;
let lastGroundPlane = null;
// Normalization shared between pointcloud, mesh, and frustums
let normCenter = null;
let normScale = 1;

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

// Apply ground plane rotation+translation to geometry in-place (before normalization)
function applyGroundPlane(geo, gp) {
  if (!gp) return;
  const { R, t } = gp;
  // p' = R @ p + t  (row-major THREE.Matrix4.set)
  const m = new THREE.Matrix4().set(
    R[0][0], R[0][1], R[0][2], t[0],
    R[1][0], R[1][1], R[1][2], t[1],
    R[2][0], R[2][1], R[2][2], t[2],
    0, 0, 0, 1
  );
  geo.applyMatrix4(m);
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
  setProgress(10, 'Loading pointcloud…');
  new PLYLoader().load(url, geo => {
    if (currentPoints) { scene.remove(currentPoints); currentPoints.geometry.dispose(); currentPoints = null; }
    applyGroundPlane(geo, gp);
    normalizeGeo(geo, true);  // fresh — sets normCenter/normScale
    const mat = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: !!geo.attributes.color,
      sizeAttenuation: true,
    });
    if (!geo.attributes.color) mat.color.set(0x2596be);
    currentPoints = new THREE.Points(geo, mat);
    scene.add(currentPoints);
    setProgress(100, `${geo.attributes.position.count.toLocaleString()} points`);
  }, xhr => {
    if (xhr.total) setProgress(Math.round(xhr.loaded / xhr.total * 90), 'Loading…');
  });
}

function loadMesh(url, gp) {
  if (!renderer) initThree();
  setProgress(10, 'Loading mesh…');
  new PLYLoader().load(url, geo => {
    if (currentMesh) { scene.remove(currentMesh); currentMesh.geometry.dispose(); currentMesh = null; }
    applyGroundPlane(geo, gp);
    normalizeGeo(geo, false);  // reuse normCenter/normScale from pointcloud
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhongMaterial({
      vertexColors: !!geo.attributes.color,
      side: THREE.DoubleSide,
      shininess: 30,
    });
    if (!geo.attributes.color) mat.color.set(0x888888);
    currentMesh = new THREE.Mesh(geo, mat);
    scene.add(currentMesh);
    setProgress(100, 'Mesh loaded');
  }, xhr => {
    if (xhr.total) setProgress(Math.round(xhr.loaded / xhr.total * 90), 'Loading mesh…');
  });
}

// Store original colors so we can reset after similarity query
let originalColors = null;

function resetPointColors() {
  if (!currentPoints || !originalColors) return;
  const geo = currentPoints.geometry;
  if (geo.attributes.color) {
    geo.attributes.color.array.set(originalColors);
    geo.attributes.color.needsUpdate = true;
  }
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
    geo.attributes.color.needsUpdate = true;
    if (statusEl) { statusEl.textContent = `✓ "${pos}"${neg ? ' − "' + neg + '"' : ''}`; statusEl.className = 'status-ok'; }
  } catch (e) {
    if (statusEl) { statusEl.textContent = String(e); statusEl.className = 'status-err'; }
  }
}

async function loadFrustums(gp) {
  if (currentFrustums) { scene.remove(currentFrustums); currentFrustums.geometry.dispose(); currentFrustums = null; }
  const resp = await fetch('/api/visualize/frustums');
  const data = await resp.json();
  if (!data.ok || !data.positions?.length) { setProgress(0, 'No frustums: ' + (data.error || 'no data')); return; }

  const positions = data.positions;
  const forward = data.forward;
  const geo = new THREE.BufferGeometry();
  const verts = [];

  positions.forEach((pos, i) => {
    // Apply ground plane transform: p' = R @ p + t
    let p = [...pos];
    if (gp) {
      const { R, t } = gp;
      p = [
        R[0][0]*pos[0] + R[0][1]*pos[1] + R[0][2]*pos[2] + t[0],
        R[1][0]*pos[0] + R[1][1]*pos[1] + R[1][2]*pos[2] + t[1],
        R[2][0]*pos[0] + R[2][1]*pos[1] + R[2][2]*pos[2] + t[2],
      ];
    }
    // Apply same normalization as pointcloud
    const px = (p[0] - (normCenter?.x || 0)) * normScale;
    const py = (p[1] - (normCenter?.y || 0)) * normScale;
    const pz = (p[2] - (normCenter?.z || 0)) * normScale;
    verts.push(px, py, pz);

    // Draw a short line in the forward direction
    let f = forward[i];
    if (gp) {
      const { R } = gp;
      f = [
        R[0][0]*f[0] + R[0][1]*f[1] + R[0][2]*f[2],
        R[1][0]*f[0] + R[1][1]*f[1] + R[1][2]*f[2],
        R[2][0]*f[0] + R[2][1]*f[1] + R[2][2]*f[2],
      ];
    }
    const s = 0.05 * normScale;
    verts.push(px + f[0]*s, py + f[1]*s, pz + f[2]*s);
  });

  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x2596be, opacity: 0.7, transparent: true });
  currentFrustums = new THREE.LineSegments(geo, mat);
  scene.add(currentFrustums);
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
      <input type="text" id="viz-pos-query" placeholder="positive query (e.g. bird)">
      <input type="text" id="viz-neg-query" placeholder="negative query (optional)">
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
    if (!e.target.checked) resetPointColors();
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
    <h4>View / Mesh</h4>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-mesh-toggle"> Show mesh</label>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-pc-toggle" checked> Show pointcloud</label>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-frustums"> Show cameras</label>
  `;

  pc.querySelector('#viz-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });
  pc.querySelector('#viz-load-btn').addEventListener('click', loadScene);

  // Mesh toggle: show mesh, hide pointcloud
  view.querySelector('#viz-mesh-toggle').addEventListener('change', async e => {
    if (e.target.checked) {
      const resp = await fetch('/api/visualize/status');
      const data = await resp.json();
      if (data.mesh_url) {
        loadMesh(data.mesh_url, data.ground_plane);
        if (currentPoints) currentPoints.visible = false;
        const pcToggle = document.getElementById('viz-pc-toggle');
        if (pcToggle) pcToggle.checked = false;
      } else {
        e.target.checked = false;
        setProgress(0, 'No mesh found');
      }
    } else {
      if (currentMesh) { scene.remove(currentMesh); currentMesh.geometry.dispose(); currentMesh = null; }
    }
  });

  // Pointcloud toggle
  view.querySelector('#viz-pc-toggle').addEventListener('change', e => {
    if (currentPoints) currentPoints.visible = e.target.checked;
    if (e.target.checked && currentMesh) {
      scene.remove(currentMesh); currentMesh.geometry.dispose(); currentMesh = null;
      const meshToggle = document.getElementById('viz-mesh-toggle');
      if (meshToggle) meshToggle.checked = false;
    }
  });

  // Camera frustums toggle
  view.querySelector('#viz-frustums').addEventListener('change', e => {
    if (e.target.checked) {
      loadFrustums(lastGroundPlane);
    } else {
      if (currentFrustums) { scene.remove(currentFrustums); currentFrustums.geometry.dispose(); currentFrustums = null; }
    }
  });

  return [pc, sem, view];
}

registerTab('visualize', {
  renderSidebar,
  onActivate() { initThree(); loadScene(); },
});
