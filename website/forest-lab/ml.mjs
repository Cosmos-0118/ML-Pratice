// Forest Lab: the machine learning, with no DOM and no dependencies.
// CART (Gini) decision trees, a bagged random forest with hard majority voting, and AdaBoost (SAMME).
// Points live in the square [-1, 1] x [-1, 1] and look like {x, y, c} with c = class index.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const clip = v => Math.max(-0.98, Math.min(0.98, v));

// ---------------------------------------------------------------- datasets
export const PRESETS = {
  moons: {
    label: "Moons", K: 2, noise: 0.1,
    make(n, noise, rng) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const c = i % 2, t = rng() * Math.PI;
        let x = c === 0 ? Math.cos(t) : 1 - Math.cos(t);
        let y = c === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        x = (x - 0.5) * 0.9 + gauss(rng) * noise; y = (y - 0.25) * 1.15 + gauss(rng) * noise;
        out.push({ x: clip(x), y: clip(y), c });
      }
      return out;
    },
  },
  circles: {
    label: "Rings", K: 2, noise: 0.06,
    make(n, noise, rng) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const c = i % 2, a = rng() * 2 * Math.PI, r = (c === 0 ? 0.34 : 0.78) + gauss(rng) * noise;
        out.push({ x: clip(r * Math.cos(a)), y: clip(r * Math.sin(a)), c });
      }
      return out;
    },
  },
  xor: {
    label: "XOR", K: 2, noise: 0.05,
    make(n, noise, rng) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const x = (rng() * 2 - 1) * 0.92, y = (rng() * 2 - 1) * 0.92;
        const c = (x > 0) !== (y > 0) ? 1 : 0;
        out.push({ x: clip(x + gauss(rng) * noise), y: clip(y + gauss(rng) * noise), c });
      }
      return out;
    },
  },
  spiral: {
    label: "Spiral", K: 3, noise: 0.05,
    make(n, noise, rng) {
      const out = [];
      for (let i = 0; i < n; i++) {
        const c = i % 3, s = Math.sqrt(rng()), a = (c * 2 * Math.PI) / 3 + s * 3.4;
        const r = 0.08 + 0.86 * s;
        out.push({ x: clip(r * Math.cos(a) + gauss(rng) * noise), y: clip(r * Math.sin(a) + gauss(rng) * noise), c });
      }
      return out;
    },
  },
  blobs: {
    label: "Blobs", K: 3, noise: 0.12,
    make(n, noise, rng) {
      const centres = [[-0.5, -0.4], [0.5, -0.3], [0.02, 0.55]], out = [];
      for (let i = 0; i < n; i++) {
        const c = i % 3, sd = 0.12 + noise;
        out.push({ x: clip(centres[c][0] + gauss(rng) * sd), y: clip(centres[c][1] + gauss(rng) * sd), c });
      }
      return out;
    },
  },
};

// A preset gives a training sample and a separate held-out sample from the same distribution.
export function makeDataset(name, noise, seed, nTrain = 150, nTest = 450) {
  const p = PRESETS[name], rng = mulberry32(seed);
  return { train: p.make(nTrain, noise, rng), test: p.make(nTest, noise, rng), K: p.K };
}

// ---------------------------------------------------------------- CART
function argmax(a) {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

// Builds one tree. `w` are sample weights (all equal for plain CART). `mtry` = features tried per split (1 or 2).
export function buildTree(pts, w, { K, maxDepth = 3, minLeaf = 1, mtry = 2, rng = Math.random }) {
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y), cs = pts.map(p => p.c);
  const feats = [xs, ys];

  function leaf(idx) {
    const counts = new Array(K).fill(0);
    let total = 0;
    for (const i of idx) { counts[cs[i]] += w[i]; total += w[i]; }
    const probs = total > 0 ? counts.map(v => v / total) : counts.map(() => 1 / K);
    return { probs, label: argmax(probs), n: idx.length };
  }

  function build(idx, depth) {
    const node = leaf(idx);
    if (depth >= maxDepth || idx.length < 2 * minLeaf || Math.max(...node.probs) > 1 - 1e-12) return node;
    let total = 0;
    for (const i of idx) total += w[i];
    const parent = 1 - node.probs.reduce((s, p) => s + p * p, 0);

    const order = rng() < 0.5 ? [0, 1] : [1, 0];
    // Features are visited in random order; `mtry` limits how many usable features are compared (1 = random-forest style).
    let best = null, visited = 0;
    for (const f of order) {
      const v = feats[f];
      const sorted = idx.slice().sort((a, b) => v[a] - v[b]);
      const left = new Array(K).fill(0), right = new Array(K).fill(0);
      let wl = 0, wr = total;
      for (const i of sorted) right[cs[i]] += w[i];
      let found = false;
      for (let k = 0; k < sorted.length - 1; k++) {
        const i = sorted[k];
        left[cs[i]] += w[i]; right[cs[i]] -= w[i]; wl += w[i]; wr -= w[i];
        const a = v[i], b = v[sorted[k + 1]];
        if (a === b || k + 1 < minLeaf || sorted.length - k - 1 < minLeaf) continue;
        let gl = 0, gr = 0;
        for (let c = 0; c < K; c++) {
          if (wl > 0) gl += (left[c] / wl) ** 2;
          if (wr > 0) gr += (Math.max(right[c], 0) / wr) ** 2;
        }
        const score = (wl * (1 - gl) + wr * (1 - gr)) / total;
        found = true;
        if (!best || score < best.score - 1e-12) best = { score, f, t: (a + b) / 2 };
      }
      if (found) visited++;
      if (visited >= mtry) break;
    }
    // Like scikit-learn, accept any valid split that does not make the node worse (XOR needs a zero-gain first split).
    if (!best || best.score > parent + 1e-9) return node;
    const li = [], ri = [];
    for (const i of idx) (feats[best.f][i] <= best.t ? li : ri).push(i);
    if (!li.length || !ri.length) return node;
    return { f: best.f, t: best.t, l: build(li, depth + 1), r: build(ri, depth + 1), probs: node.probs, label: node.label, n: idx.length };
  }

  return build(pts.map((_, i) => i), 0);
}

export function leafOf(tree, x, y) {
  let n = tree;
  while (n.l) n = (n.f === 0 ? x : y) <= n.t ? n.l : n.r;
  return n;
}
export const treeDepth = t => (t.l ? 1 + Math.max(treeDepth(t.l), treeDepth(t.r)) : 0);
export const treeSize = t => (t.l ? 1 + treeSize(t.l) + treeSize(t.r) : 1);

// ---------------------------------------------------------------- other classic learners (for the voting ensemble)
export function fitKnn(pts, k = 5) {
  const data = pts.map(p => [p.x, p.y, p.c]);
  return {
    name: `kNN (k=${k})`,
    predict(x, y) {
      const best = []; // k nearest as [distance, class], kept sorted
      for (const [px, py, c] of data) {
        const d = (px - x) ** 2 + (py - y) ** 2;
        if (best.length < k || d < best[best.length - 1][0]) {
          let i = best.length;
          while (i > 0 && best[i - 1][0] > d) i--;
          best.splice(i, 0, [d, c]);
          if (best.length > k) best.pop();
        }
      }
      const tally = new Array(Math.max(...data.map(r => r[2]), 0) + 1).fill(0);
      for (const [, c] of best) tally[c]++;
      return argmax(tally);
    },
  };
}

export function fitNaiveBayes(pts, K) {
  const stats = [];
  for (let c = 0; c < K; c++) {
    const own = pts.filter(p => p.c === c);
    if (!own.length) { stats.push(null); continue; }
    const mean = f => own.reduce((s, p) => s + p[f], 0) / own.length;
    const mx = mean("x"), my = mean("y");
    const vx = Math.max(own.reduce((s, p) => s + (p.x - mx) ** 2, 0) / own.length, 2e-3);
    const vy = Math.max(own.reduce((s, p) => s + (p.y - my) ** 2, 0) / own.length, 2e-3);
    stats.push({ mx, my, vx, vy, prior: Math.log(own.length / pts.length) });
  }
  return {
    name: "Naive Bayes",
    predict(x, y) {
      let best = 0, bs = -Infinity;
      stats.forEach((t, c) => {
        if (!t) return;
        const score = t.prior - 0.5 * (Math.log(t.vx) + Math.log(t.vy) + (x - t.mx) ** 2 / t.vx + (y - t.my) ** 2 / t.vy);
        if (score > bs) { bs = score; best = c; }
      });
      return best;
    },
  };
}

// ---------------------------------------------------------------- ensembles
export const MAX_LEARNERS = { tree: 1, forest: 100, boost: 60, vote: 3 };

// kind: "tree" | "forest" | "boost". Returns a model whose first n learners can be combined at any n.
export function trainModel(kind, pts, K, { depth = 3, learners = MAX_LEARNERS[kind], seed = 1 } = {}) {
  const rng = mulberry32(seed), n = pts.length, list = [];
  if (!n) return { kind, K, learners: list, depth };
  if (kind === "tree") {
    list.push({ tree: buildTree(pts, new Array(n).fill(1), { K, maxDepth: depth, rng }), weight: 1 });
  } else if (kind === "forest") {
    for (let m = 0; m < learners; m++) {
      const sample = [];
      for (let i = 0; i < n; i++) sample.push(pts[Math.floor(rng() * n)]); // bootstrap
      list.push({ tree: buildTree(sample, new Array(n).fill(1), { K, maxDepth: depth, mtry: 1, rng }), weight: 1 });
    }
  } else if (kind === "vote") {
    // Hard majority vote of three very different learners: boxes (tree), neighbourhoods (kNN) and ellipses (Naive Bayes).
    list.push({ name: "Decision tree", tree: buildTree(pts, new Array(n).fill(1), { K, maxDepth: depth, rng }), weight: 1 });
    const knn = fitKnn(pts, 5), nb = fitNaiveBayes(pts, K);
    list.push({ name: knn.name, predict: knn.predict, weight: 1 }, { name: nb.name, predict: nb.predict, weight: 1 });
  } else {
    let w = new Array(n).fill(1 / n);
    for (let m = 0; m < learners; m++) {
      const tree = buildTree(pts, w, { K, maxDepth: depth, rng });
      let err = 0;
      const wrong = pts.map(p => leafOf(tree, p.x, p.y).label !== p.c);
      for (let i = 0; i < n; i++) if (wrong[i]) err += w[i];
      if (list.length && err >= 1 - 1 / K) break; // no better than guessing: stop
      const alpha = err <= 1e-10 ? 4 : Math.max(Math.log((1 - err) / err) + Math.log(K - 1), 1e-3);
      list.push({ tree, weight: alpha });
      if (err <= 1e-10) break; // perfect fit: nothing left to correct
      let sum = 0;
      w = w.map((v, i) => { const u = wrong[i] ? v * Math.exp(alpha) : v; sum += u; return u; });
      w = w.map(v => v / sum);
    }
  }
  return { kind, K, learners: list, depth };
}

export function vote(model, i, x, y) {
  const l = model.learners[i];
  return l.tree ? leafOf(l.tree, x, y).label : l.predict(x, y);
}

// Share of the (weighted) vote each class gets from the first `count` learners. A lone tree reports its leaf's class mix.
export function scoresAt(model, x, y, count = model.learners.length) {
  const K = model.K, s = new Array(K).fill(0), n = Math.min(count, model.learners.length);
  if (!n) return s.map(() => 1 / K);
  if (model.kind === "tree") return leafOf(model.learners[0].tree, x, y).probs.slice();
  let total = 0;
  for (let i = 0; i < n; i++) { const wgt = model.learners[i].weight; s[vote(model, i, x, y)] += wgt; total += wgt; }
  return s.map(v => v / total);
}
export const predictAt = (model, x, y, count) => argmax(scoresAt(model, x, y, count));

export function accuracy(model, pts, count) {
  if (!pts.length || !model.learners.length) return NaN;
  let ok = 0;
  for (const p of pts) if (predictAt(model, p.x, p.y, count) === p.c) ok++;
  return ok / pts.length;
}

export function macroF1(model, pts, count, K = model.K) {
  if (!pts.length || !model.learners.length) return NaN;
  let sum = 0, used = 0;
  for (let c = 0; c < K; c++) {
    let tp = 0, fp = 0, fn = 0;
    for (const p of pts) {
      const q = predictAt(model, p.x, p.y, count);
      if (q === c && p.c === c) tp++; else if (q === c) fp++; else if (p.c === c) fn++;
    }
    if (tp + fp + fn === 0) continue;
    sum += (2 * tp) / (2 * tp + fp + fn); used++;
  }
  return used ? sum / used : NaN;
}

// ---------------------------------------------------------------- grid for the picture
// Each learner votes on every cell centre of a G x G grid once; any number of learners can then be combined cheaply.
export function cellCentre(i, G) { return -1 + ((i + 0.5) * 2) / G; }

export function computeGrid(model, G) {
  const L = model.learners.length, cells = G * G;
  const votes = new Uint8Array(L * cells);
  const tree = model.kind === "tree" ? new Float32Array(cells * model.K) : null;
  for (let j = 0; j < G; j++) {
    const y = cellCentre(j, G);
    for (let i = 0; i < G; i++) {
      const x = cellCentre(i, G), cell = j * G + i;
      for (let l = 0; l < L; l++) {
        votes[l * cells + cell] = vote(model, l, x, y);
        if (tree) { const lf = leafOf(model.learners[l].tree, x, y); for (let c = 0; c < model.K; c++) tree[cell * model.K + c] = lf.probs[c]; }
      }
    }
  }
  model.grid = { G, votes, tree };
  return model.grid;
}

// Fills `out` (Float32Array, G*G*K) with class shares for the first `count` learners.
export function gridShares(model, count, out) {
  const { G, votes, tree } = model.grid, K = model.K, cells = G * G;
  if (tree) { out.set(tree); return out; }
  const n = Math.min(count, model.learners.length);
  out.fill(0);
  let total = 0;
  for (let l = 0; l < n; l++) {
    const wgt = model.learners[l].weight, off = l * cells;
    total += wgt;
    for (let c = 0; c < cells; c++) out[c * K + votes[off + c]] += wgt;
  }
  for (let k = 0; k < out.length; k++) out[k] /= total || 1;
  return out;
}

// Train / held-out accuracy for tree depths 1..maxDepth, so the learner can see over- and under-fitting.
export function depthCurve(kind, train, test, K, { learners, seed, maxDepth = 12 }) {
  const rows = [];
  for (let d = 1; d <= maxDepth; d++) {
    const m = trainModel(kind, train, K, { depth: d, learners: kind === "tree" ? 1 : learners, seed });
    const n = m.learners.length;
    rows.push({ depth: d, train: accuracy(m, train, n), test: accuracy(m, test, n) });
  }
  return rows;
}

// ---------------------------------------------------------------- real tabular data (e.g. the Wine CSV)
// rows: [{f: number[], c: number}]. Picks two features for the map, makes a stratified 80/20 split and scales both
// features to the map using the TRAINING rows only (held-out rows are clipped to the map edge).
export function prepareReal(rows, xi, yi, seed, testShare = 0.2) {
  const rng = mulberry32(seed), byClass = new Map();
  rows.forEach(r => { if (!byClass.has(r.c)) byClass.set(r.c, []); byClass.get(r.c).push(r); });
  const train = [], test = [];
  for (const group of byClass.values()) {
    const g = group.slice();
    for (let i = g.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [g[i], g[j]] = [g[j], g[i]]; }
    const nt = Math.round(g.length * testShare);
    test.push(...g.slice(0, nt)); train.push(...g.slice(nt));
  }
  const range = fi => { const v = train.map(r => r.f[fi]); return { min: Math.min(...v), max: Math.max(...v) }; };
  const rx = range(xi), ry = range(yi);
  const scale = (v, r) => Math.max(-0.98, Math.min(0.98, -0.95 + ((v - r.min) / ((r.max - r.min) || 1)) * 1.9));
  const make = r => ({ x: scale(r.f[xi], rx), y: scale(r.f[yi], ry), c: r.c, raw: [r.f[xi], r.f[yi]] });
  return { train: train.map(make), test: test.map(make), K: byClass.size, ranges: [rx, ry] };
}
export const unscale = (u, r) => r.min + ((u + 0.95) / 1.9) * (r.max - r.min);

export function parseCsv(text) {
  const [head, ...lines] = text.trim().split(/\r?\n/);
  const names = head.split(",");
  const rows = lines.map(l => l.split(",").map(Number));
  return { names: names.slice(0, -1), rows: rows.map(r => ({ f: r.slice(0, -1), c: r[r.length - 1] })) };
}

// Describes a tree as nodes for drawing: {x (leaf order), depth, node}.
export function layoutTree(tree) {
  const nodes = [], edges = [];
  let leaf = 0;
  (function place(n, depth, parent) {
    const me = { n, depth, parent, x: 0 };
    nodes.push(me);
    if (n.l) { const a = place(n.l, depth + 1, me), b = place(n.r, depth + 1, me); me.x = (a.x + b.x) / 2; me.kids = [a, b]; }
    else me.x = leaf++;
    if (parent) edges.push([parent, me]);
    return me;
  })(tree, 0, null);
  return { nodes, edges, leaves: leaf };
}
