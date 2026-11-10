// Forest Lab: a three.js "confidence terrain" for decision trees, random forests and AdaBoost.
// Height of the map = how sure the model is; colour = the class it picks; valleys = decision boundaries.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import * as ML from "./ml.mjs";

const $ = id => document.getElementById(id);
const G = 96, HALF = 5, WORLD = HALF * 2, PR = 0.14, MAX_POINTS = 700;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const CLASS_HEX = ["#f97316", "#3b82f6", "#14b8a6"];
const CLASS_NAME = ["Orange", "Blue", "Teal"];
const KINDS = {
  tree: { dock: "One tree", depth: "Max depth", unit: "tree",
    note: "<b>CART</b> asks yes/no questions about x and y. Every question cuts the map with one straight line, so the regions are boxes. Raise the depth and the boxes get smaller, until each one hugs a single point." },
  forest: { dock: "Trees", depth: "Tree depth", unit: "trees",
    note: "<b>Random forest:</b> many deep trees, each trained on a random resample of the points. Each tree votes and the majority wins, so the jagged edges of single trees average out." },
  vote: { dock: "Voters", depth: "Tree depth", unit: "voters",
    note: "<b>Voting ensemble:</b> three very different learners vote: a decision tree (boxes), kNN (neighbourhoods) and Naive Bayes (ellipses). The majority wins, so one model's blind spot gets outvoted by the other two." },
  boost: { dock: "Rounds", depth: "Weak-learner depth", unit: "rounds",
    note: "<b>AdaBoost:</b> shallow trees trained one after another. Each round puts more weight on the points the earlier trees missed, and better trees get a louder vote." },
};

// ------------------------------------------------------------------ state
const FIXED = k => k === "tree" || k === "vote";   // models with no "grow" slider
const state = {
  real: null,
  preset: "moons", noise: 0.1, dataSeed: 7, seed: 3, baseK: 2, K: 2,
  train: [], test: [], kind: "forest", tool: 0, view3d: true, showTest: false,
  cfg: { tree: { depth: 4, n: 1 }, forest: { depth: 6, n: 60 }, boost: { depth: 2, n: 30 }, vote: { depth: 3, n: 3 } },
  model: null, hover: null, playing: false,
};
const cfg = () => state.cfg[state.kind];
const cname = k => (state.real ? `Cultivar ${k}` : CLASS_NAME[k]);
const pretty = n => n.replace(/_/g, " ");
const axisName = a => pretty(state.real.names[a === 0 ? state.real.xi : state.real.yi]);
const axisVal = (a, u) => ML.unscale(u, state.real.ranges[a]);
const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
window.forestLab = { state, ML };

// ------------------------------------------------------------------ colours from the page theme
const probe2d = document.createElement("canvas").getContext("2d");
function rgb(css) {
  probe2d.fillStyle = "#000"; probe2d.fillStyle = css;
  const h = probe2d.fillStyle;
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isDark = () => document.documentElement.getAttribute("data-theme") === "dark";
const classRGB = CLASS_HEX.map(rgb);
let neutral = [236, 236, 239];

// ------------------------------------------------------------------ renderer, camera, light
const holder = $("stage");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  if (!renderer.getContext()) throw new Error("no webgl");
} catch (e) {
  $("nogl").hidden = false;
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.setAttribute("aria-label", "3D map of the model's decisions. Click to add points.");
renderer.domElement.tabIndex = 0;
holder.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 120);
const controls = new OrbitControls(camera, renderer.domElement);
Object.assign(controls, { enableDamping: true, dampingFactor: 0.09, enablePan: false, minDistance: 4, maxDistance: 30, maxPolarAngle: 1.52, rotateSpeed: 0.7, zoomSpeed: 0.8 });
controls.target.set(0, 0.2, 0);

const ambient = new THREE.HemisphereLight(0xffffff, 0x9aa3bd, 1.15);
const sun = new THREE.DirectionalLight(0xffffff, 2.1);
sun.position.set(-5, 11, 7);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9, near: 1, far: 34 });
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03; sun.shadow.radius = 4;
scene.add(ambient, sun);

// ------------------------------------------------------------------ terrain
const cells = G * G, V = G + 1;
const cH = new Float32Array(cells), tH = new Float32Array(cells);          // unit heights (0..1): current, target
const cC = new Float32Array(cells * 3), tC = new Float32Array(cells * 3);  // colours 0..255
const delay = new Float32Array(cells);
let waveStart = 0, waveSpan = 0, hs = 0.04, hsTarget = 1.7, terrainBusy = false;
const shares = new Float32Array(cells * 3);

const texData = new Uint8Array(cells * 4);
const tex = new THREE.DataTexture(texData, G, G, THREE.RGBAFormat);
tex.colorSpace = THREE.SRGBColorSpace;
tex.magFilter = tex.minFilter = THREE.NearestFilter;
tex.generateMipmaps = false;
tex.needsUpdate = true;

const geo = new THREE.PlaneGeometry(WORLD, WORLD, G, G);
geo.rotateX(-Math.PI / 2);
geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 1, 0), 10);
const pos = geo.attributes.position;
const terrainMat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.82, metalness: 0 });
const terrain = new THREE.Mesh(geo, terrainMat);
terrain.receiveShadow = true; terrain.castShadow = true; terrain.frustumCulled = false;
scene.add(terrain);

// Skirt (the visible "slab" under the terrain) so the displaced edge never shows a gap.
const ring = [];
for (let c = 0; c <= G; c++) ring.push(c);                                   // top edge, left to right
for (let r = 1; r <= G; r++) ring.push(r * V + G);                          // right edge, downwards
for (let c = G - 1; c >= 0; c--) ring.push(G * V + c);                      // bottom edge, right to left
for (let r = G - 1; r >= 1; r--) ring.push(r * V);                          // left edge, upwards
const RN = ring.length, FLOOR = -0.42;
const skirtGeo = new THREE.BufferGeometry();
const skirtPos = new Float32Array(RN * 2 * 3), skirtIdx = [];
for (let k = 0; k < RN; k++) { const a = k * 2, b = ((k + 1) % RN) * 2; skirtIdx.push(a, a + 1, b, b, a + 1, b + 1); }
skirtGeo.setAttribute("position", new THREE.BufferAttribute(skirtPos, 3));
skirtGeo.setIndex(skirtIdx);
skirtGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 12);
const slabMat = new THREE.MeshStandardMaterial({ color: 0xdfe3ee, roughness: 0.9, side: THREE.DoubleSide });
const skirt = new THREE.Mesh(skirtGeo, slabMat);
skirt.receiveShadow = true; skirt.frustumCulled = false;
const cap = new THREE.Mesh(new THREE.PlaneGeometry(WORLD, WORLD).rotateX(Math.PI / 2), slabMat);
cap.position.y = FLOOR;
const table = new THREE.Mesh(new THREE.PlaneGeometry(80, 80).rotateX(-Math.PI / 2), new THREE.ShadowMaterial({ opacity: 0.22 }));
table.position.y = FLOOR - 0.01; table.receiveShadow = true;
scene.add(skirt, cap, table);

function cellH(i, j) { return cH[Math.max(0, Math.min(G - 1, j)) * G + Math.max(0, Math.min(G - 1, i))]; }

function writeTerrain() {
  for (let c = 0; c < cells; c++) {
    texData[c * 4] = cC[c * 3]; texData[c * 4 + 1] = cC[c * 3 + 1]; texData[c * 4 + 2] = cC[c * 3 + 2]; texData[c * 4 + 3] = 255;
  }
  tex.needsUpdate = true;
  const p = pos.array;
  for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) {
    const jA = G - r - 1, jB = G - r, iA = c - 1, iB = c;
    const h = (cellH(iA, jA) + cellH(iB, jA) + cellH(iA, jB) + cellH(iB, jB)) * 0.25;
    p[(r * V + c) * 3 + 1] = 0.05 + h * hs;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  for (let k = 0; k < RN; k++) {
    const v = ring[k] * 3;
    skirtPos[k * 6] = p[v]; skirtPos[k * 6 + 1] = p[v + 1]; skirtPos[k * 6 + 2] = p[v + 2];
    skirtPos[k * 6 + 3] = p[v]; skirtPos[k * 6 + 4] = FLOOR; skirtPos[k * 6 + 5] = p[v + 2];
  }
  skirtGeo.attributes.position.needsUpdate = true;
  skirtGeo.computeVertexNormals();
  dirty = true; pointsDirty = true;
}

// Turn class shares into target heights and colours. `origin` (data coords) is where the ripple starts.
function applyShares(wave, origin = { x: 0, y: 0 }) {
  const K = state.K, empty = !state.model || !state.model.learners.length, inv = 1 / K;
  for (let c = 0; c < cells; c++) {
    let top = 0, r = 0, g = 0, b = 0;
    for (let k = 0; k < K; k++) {
      const s = empty ? inv : shares[c * K + k];
      if (s > top) top = s;
      r += s * classRGB[k][0]; g += s * classRGB[k][1]; b += s * classRGB[k][2];
    }
    const norm = empty ? 0 : Math.max(0, (top - inv) / (1 - inv));
    const tint = 0.2 + 0.8 * Math.pow(norm, 0.75);
    tH[c] = empty ? 0 : 0.1 + 0.9 * norm;
    tC[c * 3] = neutral[0] + (r - neutral[0]) * tint;
    tC[c * 3 + 1] = neutral[1] + (g - neutral[1]) * tint;
    tC[c * 3 + 2] = neutral[2] + (b - neutral[2]) * tint;
    if (wave && !REDUCED) {
      const i = c % G, j = (c / G) | 0;
      delay[c] = Math.hypot(ML.cellCentre(i, G) - origin.x, ML.cellCentre(j, G) - origin.y) * 0.42;
    } else delay[c] = 0;
  }
  waveStart = performance.now() / 1000;
  waveSpan = wave && !REDUCED ? 0.9 : 0;
  terrainBusy = true; dirty = true;
}

function stepTerrain(dt, now) {
  const t = now / 1000, k = REDUCED ? 1 : 1 - Math.exp(-dt * 7.5);
  let moving = false;
  if (terrainBusy) {
    for (let c = 0; c < cells; c++) {
      if (t - waveStart < delay[c]) { moving = true; continue; }
      const dh = tH[c] - cH[c];
      if (Math.abs(dh) > 4e-4) { cH[c] += dh * k; moving = true; } else cH[c] = tH[c];
      for (let q = c * 3; q < c * 3 + 3; q++) {
        const dc = tC[q] - cC[q];
        if (Math.abs(dc) > 0.25) { cC[q] += dc * k; moving = true; } else cC[q] = tC[q];
      }
    }
    if (!moving) terrainBusy = false;
  }
  const dHs = hsTarget - hs;
  let hsMoving = false;
  if (Math.abs(dHs) > 1e-3) { hs += dHs * (REDUCED ? 1 : 1 - Math.exp(-dt * 5)); hsMoving = true; } else hs = hsTarget;
  if (moving || hsMoving || terrainBusy) writeTerrain();
}

// ------------------------------------------------------------------ points
const sphere = new THREE.SphereGeometry(1, 24, 16);
const pointMat = new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.05 });
const pointMesh = new THREE.InstancedMesh(sphere, pointMat, MAX_POINTS);
const haloMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide });
const haloMesh = new THREE.InstancedMesh(sphere, haloMat, MAX_POINTS);
const testGeo = new THREE.OctahedronGeometry(1);
const testMat = new THREE.MeshStandardMaterial({ roughness: 0.4, transparent: true, opacity: 0.92 });
const testMesh = new THREE.InstancedMesh(testGeo, testMat, 600);
for (const m of [pointMesh, haloMesh, testMesh]) { m.frustumCulled = false; m.count = 0; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage); scene.add(m); }
pointMesh.castShadow = true; testMesh.castShadow = true;
const dummy = new THREE.Object3D(), tmpColor = new THREE.Color();
let pointsDirty = true, colorsDirty = true, missFlags = [];

const bounceOut = x => {
  const n1 = 7.5625, d1 = 2.75;
  if (x < 1 / d1) return n1 * x * x;
  if (x < 2 / d1) return n1 * (x -= 1.5 / d1) * x + 0.75;
  if (x < 2.5 / d1) return n1 * (x -= 2.25 / d1) * x + 0.9375;
  return n1 * (x -= 2.625 / d1) * x + 0.984375;
};
const cellOf = v => Math.max(0, Math.min(G - 1, Math.floor((v + 1) / 2 * G)));
const surfaceAt = (x, y) => 0.05 + cH[cellOf(y) * G + cellOf(x)] * hs;

function updatePoints(now) {
  const pts = state.train;
  pointMesh.count = haloMesh.count = pts.length;
  let animating = false;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    let a = REDUCED ? 1 : (now - p.t0) / 650, lift = 0, s = 1;
    if (a < 1) { animating = true; if (a < 0) { s = 0; a = 0; } else { lift = 3.2 * (1 - bounceOut(a)); s = Math.min(1, a * 5); } }
    const y = surfaceAt(p.x, p.y) + PR * 0.95 + lift;
    dummy.position.set(p.x * HALF, y, -p.y * HALF);
    dummy.scale.setScalar(PR * s); dummy.updateMatrix(); pointMesh.setMatrixAt(i, dummy.matrix);
    dummy.scale.setScalar(PR * s * (missFlags[i] ? 1.8 : 1.22)); dummy.updateMatrix(); haloMesh.setMatrixAt(i, dummy.matrix);
  }
  pointMesh.instanceMatrix.needsUpdate = haloMesh.instanceMatrix.needsUpdate = true;
  if (colorsDirty) {
    const miss = cssVar("--miss") || "#e11d48";
    for (let i = 0; i < pts.length; i++) {
      tmpColor.set(CLASS_HEX[pts[i].c]); pointMesh.setColorAt(i, tmpColor);
      tmpColor.set(missFlags[i] ? miss : "#ffffff"); haloMesh.setColorAt(i, tmpColor);
    }
    if (pointMesh.instanceColor) pointMesh.instanceColor.needsUpdate = true;
    if (haloMesh.instanceColor) haloMesh.instanceColor.needsUpdate = true;
    colorsDirty = false;
  }
  // held-out points: small diamonds
  const show = state.showTest;
  testMesh.visible = show;
  if (show) {
    const ts = state.test.slice(0, 600);
    testMesh.count = ts.length;
    for (let i = 0; i < ts.length; i++) {
      const p = ts[i];
      dummy.position.set(p.x * HALF, surfaceAt(p.x, p.y) + 0.1, -p.y * HALF);
      dummy.rotation.set(0, 0, 0); dummy.scale.setScalar(0.075); dummy.updateMatrix(); testMesh.setMatrixAt(i, dummy.matrix);
      if (testDirty) { tmpColor.set(CLASS_HEX[p.c]); testMesh.setColorAt(i, tmpColor); }
    }
    testMesh.instanceMatrix.needsUpdate = true;
    if (testDirty && testMesh.instanceColor) testMesh.instanceColor.needsUpdate = true;
    testDirty = false;
  }
  pointsDirty = animating; dirty = true;
}
let testDirty = true;

// ------------------------------------------------------------------ probe + ripples
const beamMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.8, depthWrite: false });
const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 4, 12, 1, true), beamMat);
const target = new THREE.Mesh(new THREE.RingGeometry(0.14, 0.2, 48).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ transparent: true, side: THREE.DoubleSide, depthWrite: false }));
beam.visible = target.visible = false;
scene.add(beam, target);

const ripples = [];
for (let i = 0; i < 6; i++) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 48).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  m.visible = false; m.userData.t0 = 0; scene.add(m); ripples.push(m);
}
let rippleNext = 0;
function ripple(x, y, c) {
  const m = ripples[rippleNext++ % ripples.length];
  m.position.set(x * HALF, surfaceAt(x, y) + 0.06, -y * HALF);
  m.material.color.set(CLASS_HEX[c]); m.userData.t0 = performance.now(); m.visible = !REDUCED; dirty = true;
}
function stepRipples(now) {
  for (const m of ripples) {
    if (!m.visible) continue;
    const a = (now - m.userData.t0) / 700;
    if (a >= 1) { m.visible = false; continue; }
    m.scale.setScalar(0.15 + a * 1.1); m.material.opacity = (1 - a) * 0.9; dirty = true;
  }
}

// ------------------------------------------------------------------ camera poses
let tween = null, userOrbited = false, dirty = true;
function poses() {
  const fov = camera.fov * Math.PI / 180, half = HALF * 1.14;
  const d = Math.max(half / Math.tan(fov / 2), half / (Math.tan(fov / 2) * camera.aspect));
  return {
    top: new THREE.Vector3(0, d, d * 0.02),
    tilt: new THREE.Vector3(0, d * 0.86, d * 0.82),
  };
}
function flyTo(vec, ms = 1000) {
  tween = { from: camera.position.clone(), to: vec.clone(), t0: performance.now(), ms: REDUCED ? 0 : ms };
  userOrbited = false; dirty = true;
}
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
function stepCamera(now) {
  if (!tween) return;
  const a = tween.ms ? Math.min(1, (now - tween.t0) / tween.ms) : 1;
  camera.position.lerpVectors(tween.from, tween.to, ease(a));
  if (a >= 1) tween = null;
  dirty = true;
}
function setView(is3d, animate = true) {
  state.view3d = is3d;
  hsTarget = is3d ? 1.7 : 0.04;
  controls.enableRotate = is3d;
  $("v3d").setAttribute("aria-pressed", is3d);
  $("v2d").setAttribute("aria-pressed", !is3d);
  const p = poses()[is3d ? "tilt" : "top"];
  if (animate) flyTo(p, 1100); else { camera.position.copy(p); tween = null; }
  dirty = true;
}
controls.addEventListener("start", () => { tween = null; userOrbited = true; });
controls.addEventListener("change", () => { dirty = true; });

function resize() {
  const w = holder.clientWidth, h = holder.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  if (!userOrbited && !tween) camera.position.copy(poses()[state.view3d ? "tilt" : "top"]);
  dirty = true;
}
new ResizeObserver(resize).observe(holder);

// ------------------------------------------------------------------ theme
function applyTheme() {
  const dark = isDark();
  neutral = dark ? [44, 44, 48] : [236, 236, 239];
  slabMat.color.set(dark ? "#18181b" : "#e2e2e6");
  ambient.intensity = dark ? 1.9 : 1.15;
  sun.intensity = dark ? 1.9 : 2.1;
  beamMat.color.set(dark ? "#ffffff" : "#11151f");
  colorsDirty = true;
  if (state.model) applyShares(false);
  drawCurve(); dirty = true;
}
new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

// ------------------------------------------------------------------ modelling
let compareRows = [], compareKey = "", curveRows = [], curveKey = "", curveTimer = 0, retrainQueued = null;

function nMax() { return Math.max(1, state.model ? state.model.learners.length : 1); }
function recomputeK() { state.K = Math.max(state.baseK, ...state.train.map(p => p.c + 1)); }

function retrain(wave = true, origin) {
  recomputeK();
  const c = cfg();
  state.model = ML.trainModel(state.kind, state.train, state.K, { depth: c.depth, learners: ML.MAX_LEARNERS[state.kind], seed: state.seed });
  if (state.model.learners.length) ML.computeGrid(state.model, G);
  c.n = Math.max(1, Math.min(c.n, nMax()));
  refresh(wave, origin);
  scheduleCurve();
}
function queueRetrain(wave = true, origin) {
  retrainQueued = { wave, origin };
  requestAnimationFrame(() => { if (retrainQueued) { const q = retrainQueued; retrainQueued = null; retrain(q.wave, q.origin); } });
}

function paintRanges() {
  for (const r of document.querySelectorAll('input[type=range]')) r.style.setProperty("--p", ((r.value - r.min) / ((r.max - r.min) || 1)) * 100 + "%");
}
document.addEventListener("input", e => { if (e.target.type === "range") paintRanges(); });
const pct = v => (Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "–");

function refresh(wave = false, origin) {
  const m = state.model, K = state.K, n = cfg().n, has = m && m.learners.length;
  if (has) ML.gridShares(m, n, shares); else shares.fill(1 / K);
  applyShares(wave, origin);

  missFlags = state.train.map(p => has && ML.predictAt(m, p.x, p.y, n) !== p.c);
  colorsDirty = true; pointsDirty = true;

  const tr = has ? ML.accuracy(m, state.train, n) : NaN;
  const te = has && state.test.length ? ML.accuracy(m, state.test, n) : NaN;
  const f1 = has && state.test.length ? ML.macroF1(m, state.test, n, K) : NaN;
  $("sTrain").textContent = pct(tr);
  $("sTest").textContent = pct(te);
  $("sF1").textContent = Number.isFinite(f1) ? f1.toFixed(3) : "–";
  $("sMiss").textContent = has ? `${missFlags.filter(Boolean).length} / ${state.train.length}` : "–";
  insight(tr, te);

  const dockOff = FIXED(state.kind);
  $("nSlider").max = nMax(); $("nSlider").value = n; $("nSlider").disabled = dockOff;
  document.querySelector(".fl-dock").dataset.off = dockOff;
  $("nOut").textContent = dockOff ? (state.kind === "vote" ? "3 voters" : "1 tree") : `${n} / ${nMax()}`;
  $("nLabel").textContent = KINDS[state.kind].dock;
  $("play").disabled = dockOff;

  const counts = new Array(K).fill(0);
  for (const p of state.train) counts[p.c]++;
  $("legend").innerHTML = counts.map((c, k) => `<span><i style="background:${CLASS_HEX[k]}"></i>${cname(k)} · ${c}</span>`).join("")
    + (state.real ? `<span class="ax">x · ${axisName(0)}</span><span class="ax">y · ${axisName(1)}</span>` : "");
  if (treeFor !== m) { treeFor = m; drawTree(); }
  updateBooth();
  paintRanges();
  dirty = true;
}

function insight(tr, te) {
  const el = $("insight"), c = cfg();
  if (!Number.isFinite(tr)) { el.textContent = "Drop some points or pick a dataset to get started."; return; }
  if (!Number.isFinite(te)) { el.textContent = "There is no held-out set for hand-drawn data, so only training accuracy is shown."; return; }
  const gap = tr - te;
  if (gap > 0.1) el.innerHTML = `<b>Over-fitting:</b> training beats held-out by ${(gap * 100).toFixed(0)} points. The model is memorising noise${state.kind === "forest" ? "" : "; try a smaller depth"}.`;
  else if (tr < 0.8 && te < 0.8) el.innerHTML = `<b>Under-fitting:</b> both scores are low at depth ${c.depth}. The model is too simple for this shape, so raise the depth${state.kind === "tree" ? "" : " or the learner count"}.`;
  else el.innerHTML = "<b>Healthy fit:</b> training and held-out accuracy are close, so the model has learned the shape rather than the noise.";
}

// ------------------------------------------------------------------ voting booth
function updateBooth() {
  const m = state.model, h = state.hover, booth = $("booth");
  const ok = h && m && m.learners.length;
  booth.hidden = !ok; $("boothHint").hidden = !!ok;
  $("live").textContent = !ok ? "Hover the map" : state.real ? `${fmt(axisVal(0, h.x))} · ${fmt(axisVal(1, h.y))}` : `x ${h.x.toFixed(2)}  y ${h.y.toFixed(2)}`;
  $("live").classList.toggle("on", !!ok);
  beam.visible = target.visible = !!ok;
  highlightPath(ok ? h : null);
  if (!ok) { dirty = true; return; }
  const n = cfg().n, s = ML.scoresAt(m, h.x, h.y, n), win = s.indexOf(Math.max(...s));
  $("verdict").innerHTML = `<i style="background:${CLASS_HEX[win]}"></i>${cname(win)}<small>${(s[win] * 100).toFixed(0)}% sure</small>`;
  const ballots = $("ballots");
  ballots.classList.toggle("named", m.kind === "vote");
  if (m.kind === "tree") ballots.replaceChildren();
  else if (m.kind === "vote") {
    ballots.innerHTML = m.learners.map((l, i) => { const c = ML.vote(m, i, h.x, h.y); return `<div><i style="background:${CLASS_HEX[c]}"></i><span>${l.name}</span><em>${cname(c)}</em></div>`; }).join("");
  } else {
    const wmax = Math.max(...m.learners.slice(0, n).map(l => l.weight));
    if (ballots.children.length !== n || ballots.classList.contains("named")) {   // rebuild (with the pop-in) only when the number of voters changes
      ballots.replaceChildren(...Array.from({ length: n }, (_, i) => { const b = document.createElement("i"); b.style.setProperty("--k", i); return b; }));
    }
    for (let i = 0; i < n; i++) {
      const b = ballots.children[i];
      b.style.background = CLASS_HEX[ML.vote(m, i, h.x, h.y)];
      b.style.opacity = m.kind === "boost" ? 0.35 + 0.65 * (m.learners[i].weight / wmax) : 1;
    }
  }
  $("tally").innerHTML = s.map((v, k) => {
    const label = m.kind === "forest" ? `${Math.round(v * n)} / ${n}` : m.kind === "vote" ? `${Math.round(v * 3)} / 3` : `${(v * 100).toFixed(0)}%`;
    return `<div><i style="background:${CLASS_HEX[k]}"></i><span><b style="width:${v * 100}%;background:${CLASS_HEX[k]}"></b></span><em>${label}</em></div>`;
  }).join("");
  const y = surfaceAt(h.x, h.y);
  beam.position.set(h.x * HALF, y + 2, -h.y * HALF);
  target.position.set(h.x * HALF, y + 0.05, -h.y * HALF);
  beam.material.color.set(CLASS_HEX[win]);
  target.material.color.set(CLASS_HEX[win]);
  dirty = true;
}

// ------------------------------------------------------------------ CART diagram (single-tree mode)
let treeFor = null, treeNodes = new Map();
const svgNS = "http://www.w3.org/2000/svg";
function splitText(n) {
  if (state.real) { const a = n.f; return { name: axisName(a), val: fmt(axisVal(a, n.t)) }; }
  return { name: n.f === 0 ? "x" : "y", val: n.t.toFixed(2) };
}
function drawTree() {
  const sec = $("treeSec"), m = state.model, box = $("treeBox");
  const show = state.kind === "tree" && m && m.learners.length;
  sec.hidden = !show; treeNodes = new Map();
  if (!show) { box.replaceChildren(); return; }
  const root = m.learners[0].tree, { nodes, edges, leaves } = ML.layoutTree(root);
  const DX = 50, DY = 58, PX = 34, PY = 22, depth = Math.max(...nodes.map(n => n.depth));
  const W = Math.max(leaves, 1) * DX + PX * 2 - DX + 56, H = (depth + 1) * DY + PY;
  const px = n => PX + n.x * DX + 28, py = n => PY + n.depth * DY;
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "The decision tree: each box asks a yes/no question, each coloured leaf gives a class.");
  let html = "";
  for (const [p, c] of edges) html += `<path class="te" data-id="${nodes.indexOf(c)}" d="M${px(p)} ${py(p) + 12}C${px(p)} ${py(p) + 34} ${px(c)} ${py(c) - 30} ${px(c)} ${py(c) - 12}"/>`;
  nodes.forEach((nd, i) => {
    const n = nd.n, x = px(nd), y = py(nd);
    if (n.l) {
      const t = splitText(n), label = `${t.name.length > 11 ? t.name.slice(0, 10) + "…" : t.name} ≤ ${t.val}`, w = label.length * 5.9 + 14;
      html += `<g class="tn" data-id="${i}"><title>${t.name} ≤ ${t.val}? yes goes left, no goes right (${n.n} points)</title><rect x="${x - w / 2}" y="${y - 12}" width="${w}" height="24" rx="7"/><text x="${x}" y="${y + 4}" text-anchor="middle">${label}</text></g>`;
    } else {
      html += `<g class="tl" data-id="${i}"><title>${cname(n.label)}: ${n.n} points, ${(n.probs[n.label] * 100).toFixed(0)}% ${cname(n.label)}</title><rect x="${x - 20}" y="${y - 12}" width="40" height="24" rx="7" fill="${CLASS_HEX[n.label]}"/><text x="${x}" y="${y + 4}" text-anchor="middle">${n.n}</text></g>`;
    }
  });
  svg.innerHTML = html; box.replaceChildren(svg);
  nodes.forEach((nd, i) => treeNodes.set(nd.n, { el: svg.querySelector(`g[data-id="${i}"]`), edge: svg.querySelector(`path[data-id="${i}"]`) }));
  $("treeMeta").textContent = `${leaves} leaves · depth ${depth}`;
}
function highlightPath(h) {
  for (const { el, edge } of treeNodes.values()) { el.classList.remove("on"); if (edge) edge.classList.remove("on"); }
  if (!h || state.kind !== "tree" || !state.model || !state.model.learners.length) return;
  let n = state.model.learners[0].tree;
  for (;;) {
    const t = treeNodes.get(n); if (t) { t.el.classList.add("on"); if (t.edge) t.edge.classList.add("on"); }
    if (!n.l) break;
    n = (n.f === 0 ? h.x : h.y) <= n.t ? n.l : n.r;
  }
}

// ------------------------------------------------------------------ curve (accuracy vs depth)
const curveEl = $("curve"), cg = curveEl.getContext("2d");
const hashPts = a => a.length + ":" + a.reduce((s, p) => s + p.x * 7.1 + p.y * 13.3 + p.c * 3.7, 0).toFixed(4);
function scheduleCurve() {
  clearTimeout(curveTimer);
  const c = cfg(), learners = Math.min(c.n, 40);
  const key = [state.kind, state.seed, learners, hashPts(state.train), state.test.length].join("|");
  const ckey = [state.seed, hashPts(state.train), state.test.length, ...Object.values(state.cfg).map(v => v.depth + "/" + v.n)].join("|");
  if (key === curveKey && ckey === compareKey) { drawCurve(); drawCompare(); return; }
  curveTimer = setTimeout(() => {
    if (key !== curveKey) {
      curveKey = key;
      curveRows = state.train.length ? ML.depthCurve(state.kind, state.train, state.test, state.K, { learners, seed: state.seed }) : [];
    }
    if (ckey !== compareKey) { compareKey = ckey; computeCompare(); }
    drawCurve(); drawCompare();
  }, 160);
}
// Every model at its own current settings, side by side (a classroom comparison, not a way to pick a winner).
function computeCompare() {
  compareRows = [];
  if (!state.train.length) return;
  for (const kind of ["tree", "forest", "boost", "vote"]) {
    const c = state.cfg[kind];
    const m = ML.trainModel(kind, state.train, state.K, { depth: c.depth, learners: c.n, seed: state.seed });
    const n = m.learners.length;
    compareRows.push({ kind, name: { tree: "Single tree", forest: "Random forest", boost: "AdaBoost", vote: "Voting" }[kind], train: ML.accuracy(m, state.train, n), test: ML.accuracy(m, state.test, n) });
  }
  const counts = new Array(state.K).fill(0);
  state.train.forEach(p => counts[p.c]++);
  const major = counts.indexOf(Math.max(...counts));
  compareRows.push({ kind: "", name: "Always the majority", train: counts[major] / state.train.length, test: state.test.length ? state.test.filter(p => p.c === major).length / state.test.length : NaN });
}
function drawCompare() {
  $("compareBody").innerHTML = compareRows.map(r => `<tr${r.kind ? ` data-kind="${r.kind}"` : ' class="base"'}${r.kind === state.kind ? ' aria-current="true"' : ""}><td>${r.name}</td><td>${pct(r.train)}</td><td>${pct(r.test)}</td></tr>`).join("");
}
function drawCurve() {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = curveEl.clientWidth || 320, H = 150;
  if (curveEl.width !== W * dpr) { curveEl.width = W * dpr; curveEl.height = H * dpr; }
  cg.setTransform(dpr, 0, 0, dpr, 0, 0); cg.clearRect(0, 0, W, H);
  const text = cssVar("--muted"), line = cssVar("--border-strong"), ink = cssVar("--text"), accent = cssVar("--accent");
  cg.font = "11px " + (cssVar("--mono") || "monospace"); cg.textBaseline = "middle";
  if (!curveRows.length) { cg.fillStyle = text; cg.textAlign = "center"; cg.fillText("Accuracy vs depth appears here", W / 2, H / 2); return; }
  const L = 34, R = 10, T = 12, B = 24, pw = W - L - R, ph = H - T - B;
  const vals = curveRows.flatMap(r => [r.train, r.test]).filter(Number.isFinite);
  const lo = Math.min(0.9, Math.floor(Math.min(...vals) * 10) / 10 - 0.0), X = d => L + ((d - 1) / 11) * pw, Y = v => T + (1 - (v - lo) / (1 - lo)) * ph;
  cg.strokeStyle = line; cg.lineWidth = 1; cg.fillStyle = text; cg.textAlign = "right";
  for (const v of [lo, (lo + 1) / 2, 1]) { cg.beginPath(); cg.moveTo(L, Y(v)); cg.lineTo(W - R, Y(v)); cg.globalAlpha = 0.5; cg.stroke(); cg.globalAlpha = 1; cg.fillText((v * 100).toFixed(0) + "%", L - 6, Y(v)); }
  cg.textAlign = "center";
  for (const d of [1, 4, 8, 12]) cg.fillText(d, X(d), H - 9);
  const cur = cfg().depth;
  cg.setLineDash([3, 3]); cg.strokeStyle = ink; cg.globalAlpha = 0.5; cg.beginPath(); cg.moveTo(X(cur), T); cg.lineTo(X(cur), T + ph); cg.stroke(); cg.setLineDash([]); cg.globalAlpha = 1;
  const draw = (key, colour, dashed) => {
    const rows = curveRows.filter(r => Number.isFinite(r[key]));
    if (!rows.length) return;
    cg.strokeStyle = colour; cg.fillStyle = colour; cg.lineWidth = 2.2; cg.lineJoin = "round"; cg.setLineDash(dashed ? [5, 4] : []);
    cg.beginPath(); rows.forEach((r, i) => (i ? cg.lineTo(X(r.depth), Y(r[key])) : cg.moveTo(X(r.depth), Y(r[key])))); cg.stroke(); cg.setLineDash([]);
    for (const r of rows) { cg.beginPath(); cg.arc(X(r.depth), Y(r[key]), r.depth === cur ? 4.5 : 2.2, 0, 6.2832); cg.fill(); }
  };
  draw("train", accent, false); draw("test", ink, true);
  cg.textAlign = "right"; cg.fillStyle = accent; cg.fillText("● train", W - R - 74, T + ph - 8);
  cg.fillStyle = ink; cg.fillText("◦ held-out", W - R - 2, T + ph - 8);
}
curveEl.addEventListener("click", e => {
  const r = curveEl.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * (curveEl.clientWidth || 320);
  const d = Math.max(1, Math.min(12, Math.round(1 + (x - 34) / ((curveEl.clientWidth || 320) - 44) * 11)));
  setDepth(d);
});
function setDepth(d) { cfg().depth = d; $("depth").value = d; $("depthOut").textContent = d; paintRanges(); drawCurve(); queueRetrain(false); }

// ------------------------------------------------------------------ data
let wineCache = null, featSel = { xi: -1, yi: -1 };
function setRealUI(on) {
  $("stage").parentElement.dataset.real = on;
  $("noiseField").hidden = on; $("features").hidden = !on;
  $("hint").textContent = on ? "Real Wine measurements · hover to inspect any spot · pick the two features below" : "Click to drop a point · drag to orbit · scroll to zoom · shift-click a point to remove it";
  if (on && state.tool === "erase") setTool(0);
}
async function loadReal(reshuffle = false) {
  if (reshuffle) state.dataSeed = (state.dataSeed * 9301 + 49297) % 233280;
  if (!wineCache) {
    try { wineCache = ML.parseCsv(await (await fetch("../data/wine.csv")).text()); }
    catch (e) { $("insight").textContent = "Could not load the Wine data, showing a toy dataset instead."; return loadPreset("moons"); }
    featSel = { xi: wineCache.names.indexOf("flavanoids"), yi: wineCache.names.indexOf("color_intensity") };
    const opts = wineCache.names.map((n, i) => `<option value="${i}">${pretty(n)}</option>`).join("");
    $("featX").innerHTML = opts; $("featY").innerHTML = opts;
  }
  $("featX").value = featSel.xi; $("featY").value = featSel.yi;
  const ds = ML.prepareReal(wineCache.rows, featSel.xi, featSel.yi, state.dataSeed), now = performance.now();
  state.real = { names: wineCache.names, xi: featSel.xi, yi: featSel.yi, ranges: ds.ranges };
  state.preset = "wine"; state.train = ds.train.map((p, i) => ({ ...p, t0: now + i * 6 })); state.test = ds.test; testDirty = true; state.baseK = ds.K;
  setRealUI(true);
  for (const b of document.querySelectorAll("#presets .fl-chip")) b.setAttribute("aria-pressed", b.dataset.name === "wine");
  retrain(true);
}
function loadPreset(name, reshuffle = false) {
  if (name === "wine") return loadReal(reshuffle);
  if (state.real) { state.real = null; setRealUI(false); }
  if (reshuffle) state.dataSeed = (state.dataSeed * 9301 + 49297) % 233280;
  state.preset = name;
  const ds = ML.makeDataset(name, state.noise, state.dataSeed);
  const now = performance.now();
  state.train = ds.train.map((p, i) => ({ ...p, t0: now + i * 6 }));
  state.test = ds.test; testDirty = true;
  state.baseK = ds.K;
  for (const b of document.querySelectorAll("#presets .fl-chip")) b.setAttribute("aria-pressed", b.dataset.name === name);
  retrain(true);
  $("hint").classList.remove("quiet");
}
function addPoint(x, y) {
  if (state.real) return;
  if (state.train.length >= MAX_POINTS - 1) return;
  const c = state.tool;
  state.train.push({ x, y, c, t0: performance.now() });
  ripple(x, y, c);
  queueRetrain(true, { x, y });
}
function erasePoint(x, y) {
  if (state.real) return;
  let best = -1, bd = 0.09;
  state.train.forEach((p, i) => { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = i; } });
  if (best < 0) return;
  state.train.splice(best, 1);
  queueRetrain(true, { x, y });
}

// ------------------------------------------------------------------ pointer handling
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(terrain, false)[0];
  return hit ? { x: Math.max(-1, Math.min(1, hit.point.x / HALF)), y: Math.max(-1, Math.min(1, -hit.point.z / HALF)) } : null;
}
let down = null;
const dom = renderer.domElement;
dom.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button, moved: false }; });
addEventListener("pointermove", e => {
  if (down && !down.moved && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) { down.moved = true; holder.classList.add("grab"); }
});
addEventListener("pointerup", e => {
  holder.classList.remove("grab");
  const d = down; down = null;
  if (!d || d.moved || d.button !== 0 || e.target !== dom || performance.now() - d.t > 700) return;
  const p = pick(e);
  if (!p) return;
  if (e.shiftKey || state.tool === "erase") erasePoint(p.x, p.y); else addPoint(p.x, p.y);
  state.hover = p; $("hint").classList.add("quiet");
});
let moveEv = null, moveRaf = 0;
dom.addEventListener("pointermove", e => {
  if (down && down.moved) return;
  moveEv = e;
  if (!moveRaf) moveRaf = requestAnimationFrame(() => { moveRaf = 0; const ev = moveEv; moveEv = null; if (ev) { state.hover = pick(ev); updateBooth(); } });
});
dom.addEventListener("pointerleave", () => { if (!down) { state.hover = null; updateBooth(); } });
dom.addEventListener("contextmenu", e => e.preventDefault());

// ------------------------------------------------------------------ UI wiring
const THUMBS = {
  moons: `<path d="M3 14a8 8 0 0 1 16 0" stroke="${CLASS_HEX[0]}"/><path d="M9 15a8 8 0 0 0 16 0" stroke="${CLASS_HEX[1]}"/>`,
  circles: `<circle cx="14" cy="14" r="3.5" stroke="${CLASS_HEX[0]}"/><circle cx="14" cy="14" r="10" stroke="${CLASS_HEX[1]}"/>`,
  xor: `<g stroke="none"><circle cx="8" cy="8" r="3.4" fill="${CLASS_HEX[0]}"/><circle cx="20" cy="20" r="3.4" fill="${CLASS_HEX[0]}"/><circle cx="20" cy="8" r="3.4" fill="${CLASS_HEX[1]}"/><circle cx="8" cy="20" r="3.4" fill="${CLASS_HEX[1]}"/></g>`,
  spiral: [0, 120, 240].map((r, k) => `<path d="M14 14c1-6 8-7 10-2" stroke="${CLASS_HEX[k]}" transform="rotate(${r} 14 14)"/>`).join(""),
  wine: `<g stroke="none"><circle cx="6" cy="20" r="2.6" fill="${CLASS_HEX[0]}"/><circle cx="10" cy="16" r="2.6" fill="${CLASS_HEX[0]}"/><circle cx="14" cy="13" r="2.6" fill="${CLASS_HEX[1]}"/><circle cx="18" cy="9" r="2.6" fill="${CLASS_HEX[1]}"/><circle cx="22" cy="6" r="2.6" fill="${CLASS_HEX[2]}"/><circle cx="20" cy="15" r="2.6" fill="${CLASS_HEX[2]}"/></g>`,
  blobs: `<g stroke="none"><circle cx="8" cy="9" r="4" fill="${CLASS_HEX[0]}"/><circle cx="20" cy="9" r="4" fill="${CLASS_HEX[1]}"/><circle cx="14" cy="21" r="4" fill="${CLASS_HEX[2]}"/></g>`,
};
const ERASER = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m9.5 3 4 4-5.5 5.5H5L2.5 10 9.5 3ZM6 12.5h7.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
function build() {
  const entries = [...Object.entries(ML.PRESETS).map(([k, p]) => [k, p.label]), ["wine", "Wine"]];
  $("presets").innerHTML = entries.map(([k, label]) =>
    `<button type="button" class="fl-chip${k === "wine" ? " real" : ""}" data-name="${k}" aria-pressed="${k === state.preset}"><svg viewBox="0 0 28 28" fill="none" stroke-width="3" stroke-linecap="round" aria-hidden="true">${THUMBS[k]}</svg>${label}${k === "wine" ? "<small>real data</small>" : ""}</button>`).join("");
  $("brushes").innerHTML = CLASS_HEX.map((h, k) =>
    `<button type="button" class="fl-chip" data-tool="${k}" aria-pressed="${k === 0}" title="Paint ${CLASS_NAME[k]} (${k + 1})"><i class="sw" style="background:${h}"></i><span class="k">${k + 1}</span></button>`).join("")
    + `<button type="button" class="fl-chip" data-tool="erase" aria-pressed="false" title="Erase points (E)" aria-label="Erase">${ERASER}</button>`;
}
build();

$("presets").addEventListener("click", e => {
  const b = e.target.closest("[data-name]"); if (!b) return;
  const name = b.dataset.name, again = name === state.preset;
  if (!again && name !== "wine") { state.noise = ML.PRESETS[name].noise; $("noise").value = state.noise; $("noiseOut").textContent = state.noise.toFixed(2); }
  loadPreset(name, again);
});
$("shuffle").addEventListener("click", () => loadPreset(state.preset || "moons", true));
for (const [id, key, other] of [["featX", "xi", "yi"], ["featY", "yi", "xi"]]) {
  $(id).addEventListener("change", e => {
    featSel[key] = +e.target.value;
    if (featSel.xi === featSel.yi) featSel[other] = (featSel[key] + 1) % wineCache.names.length;
    loadReal(false);
  });
}
$("noise").addEventListener("input", e => { state.noise = +e.target.value; $("noiseOut").textContent = state.noise.toFixed(2); loadPreset(state.preset); });
function setTool(t) {
  if (state.real && t === "erase") return;
  state.tool = t;
  for (const b of document.querySelectorAll("#brushes .fl-chip")) b.setAttribute("aria-pressed", String(b.dataset.tool) === String(t));
  holder.classList.toggle("erase", t === "erase");
}
$("brushes").addEventListener("click", e => { const b = e.target.closest("[data-tool]"); if (b) setTool(b.dataset.tool === "erase" ? "erase" : +b.dataset.tool); });
$("clear").addEventListener("click", () => {
  state.train = []; state.test = []; state.baseK = 2; state.preset = ""; testDirty = true;
  for (const b of document.querySelectorAll("#presets .fl-chip")) b.setAttribute("aria-pressed", "false");
  retrain(true);
});
function setKind(kind) {
  stopPlay();
  state.kind = kind;
  for (const b of document.querySelectorAll("#models button")) b.setAttribute("aria-pressed", b.dataset.kind === kind);
  const c = cfg();
  $("depth").value = c.depth; $("depthOut").textContent = c.depth;
  $("depthLabel").textContent = KINDS[kind].depth;
  $("modelNote").innerHTML = KINDS[kind].note;
  $("reroll").hidden = FIXED(kind);
  retrain(true);
}
$("models").addEventListener("click", e => { const b = e.target.closest("[data-kind]"); if (b && b.dataset.kind !== state.kind) setKind(b.dataset.kind); });
$("depth").addEventListener("input", e => setDepth(+e.target.value));
$("reroll").addEventListener("click", () => { state.seed = (state.seed * 1103515245 + 12345) % 2147483647; retrain(true); });
$("showTest").addEventListener("change", e => { state.showTest = e.target.checked; testDirty = true; pointsDirty = true; dirty = true; });
$("nSlider").addEventListener("input", e => { stopPlay(); setN(+e.target.value); });
function setN(n) { cfg().n = Math.max(1, Math.min(nMax(), Math.round(n))); refresh(false); scheduleCurve(); }
$("compareBody").addEventListener("click", e => { const r = e.target.closest("[data-kind]"); if (r && r.dataset.kind !== state.kind) setKind(r.dataset.kind); });
$("v3d").addEventListener("click", () => setView(true));
$("v2d").addEventListener("click", () => setView(false));

// "Grow the ensemble" playback
let playRaf = 0, playFrom = 0, playT0 = 0;
function stopPlay() { state.playing = false; cancelAnimationFrame(playRaf); $("play").dataset.playing = "false"; }
function startPlay() {
  if (FIXED(state.kind)) return;
  state.playing = true; $("play").dataset.playing = "true";
  if (cfg().n >= nMax()) setN(1);
  playFrom = cfg().n; playT0 = performance.now();
  const speed = nMax() / (REDUCED ? 0.5 : 5);
  const tick = now => {
    if (!state.playing) return;
    const n = Math.min(nMax(), Math.floor(playFrom + ((now - playT0) / 1000) * speed) + 1);
    if (n !== cfg().n) setN(n);
    if (n >= nMax()) { stopPlay(); return; }
    playRaf = requestAnimationFrame(tick);
  };
  playRaf = requestAnimationFrame(tick);
}
$("play").addEventListener("click", () => (state.playing ? stopPlay() : startPlay()));

addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && e.key !== " ") return;
  if (e.key >= "1" && e.key <= "3") { if (!state.real) setTool(+e.key - 1); }
  else if (e.key === "e" || e.key === "E") setTool("erase");
  else if (e.key === "v" || e.key === "V") setView(!state.view3d);
  else if (e.key === " " && !/^(BUTTON|INPUT)$/.test(e.target.tagName)) { e.preventDefault(); state.playing ? stopPlay() : startPlay(); }
});

// ------------------------------------------------------------------ main loop (renders only when something changed)
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  stepCamera(now);
  if (controls.update()) dirty = true;
  stepTerrain(dt, now);
  if (pointsDirty) updatePoints(now);
  stepRipples(now);
  if (dirty) { dirty = false; renderer.render(scene, camera); }
  requestAnimationFrame(frame);
}

applyTheme();
for (let c = 0; c < cells; c++) { cC[c * 3] = neutral[0]; cC[c * 3 + 1] = neutral[1]; cC[c * 3 + 2] = neutral[2]; }
$("modelNote").innerHTML = KINDS[state.kind].note;
$("depth").value = cfg().depth; $("depthOut").textContent = cfg().depth; $("depthLabel").textContent = KINDS[state.kind].depth;
$("noise").value = state.noise;
resize();
setView(true, false);
camera.lookAt(controls.target);
const qs = new URLSearchParams(location.search);
if (qs.get("embed")) document.body.classList.add("embed");
const startKind = qs.get("model") || (qs.get("q") === "1" ? "tree" : "forest");
if (startKind !== state.kind && KINDS[startKind]) setKind(startKind);
loadPreset(qs.get("data") === "wine" ? "wine" : "moons");
requestAnimationFrame(frame);
