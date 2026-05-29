import { state, registerTab, setProgress } from './state.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/PLYLoader.js';

let renderer, scene, camera, controls, currentPoints;

function initThree() {
  const canvas = document.getElementById('three-canvas');
  if (!canvas || renderer) return;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0a0a0f);

  scene = new THREE.Scene();
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

function loadPLY(url, pointSize) {
  pointSize = pointSize || 0.004;
  if (!renderer) initThree();
  setProgress(10, 'Loading pointcloud…');
  new PLYLoader().load(url, geo => {
    if (currentPoints) { scene.remove(currentPoints); currentPoints.geometry.dispose(); }
    geo.computeBoundingBox();
    const center = new THREE.Vector3();
    geo.boundingBox.getCenter(center);
    geo.translate(-center.x, -center.y, -center.z);
    const size = new THREE.Vector3();
    geo.boundingBox.getSize(size);
    const scale = 2 / Math.max(size.x, size.y, size.z);
    geo.scale(scale, scale, scale);
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

async function loadScene() {
  if (!state.outputDir) return;
  const resp = await fetch('/api/visualize/status');
  const data = await resp.json();
  if (!data.ok) { setProgress(0, data.error || 'No session'); return; }

  // Sync creator dropdown to auto-detected backend
  if (data.creator) {
    state.creator = data.creator;
    const sel = document.getElementById('viz-creator');
    if (sel) {
      // Rebuild options from available_backends if provided
      if (data.available_backends?.length) {
        sel.innerHTML = '';
        data.available_backends.forEach(b => {
          const o = document.createElement('option');
          o.value = b; o.textContent = b;
          if (b === data.creator) o.selected = true;
          sel.appendChild(o);
        });
      } else {
        sel.value = data.creator;
      }
    }
  }

  if (data.ply_url) {
    loadPLY(data.ply_url);
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
    <select id="viz-creator">
      <option value="vggtx">vggtx</option>
      <option value="mapanything">mapanything</option>
      <option value="vggt_omega">vggt_omega</option>
    </select>
    <button class="primary" id="viz-load-btn" style="margin-top:4px">Load scene</button>
  `;
  const sem = document.createElement('div');
  sem.className = 'sidebar-section';
  sem.innerHTML = `
    <h4>Semantic methods</h4>
    <select id="viz-extractor">
      <option value="dinov2">DINOv2</option>
      <option value="sam">SAM</option>
    </select>
  `;
  const view = document.createElement('div');
  view.className = 'sidebar-section';
  view.innerHTML = `
    <h4>View / Mesh</h4>
    <label style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="viz-frustums"> Show frustums</label>
    <label>Point size: <span id="viz-pt-val">4</span></label>
    <input type="range" id="viz-pt-size" min="1" max="20" value="4">
  `;

  pc.querySelector('#viz-creator').addEventListener('change', e => {
    state.creator = e.target.value;
    fetch('/api/session/update', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({creator: e.target.value}) });
  });
  pc.querySelector('#viz-load-btn').addEventListener('click', loadScene);
  view.querySelector('#viz-pt-size').addEventListener('input', e => {
    view.querySelector('#viz-pt-val').textContent = e.target.value;
    if (currentPoints) currentPoints.material.size = parseFloat(e.target.value) * 0.001;
  });

  return [pc, sem, view];
}

registerTab('visualize', {
  renderSidebar,
  onActivate() { initThree(); loadScene(); },
});
