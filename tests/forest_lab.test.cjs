const assert = require("node:assert/strict");
const { join } = require("node:path");
const { pathToFileURL } = require("node:url");
const { test } = require("node:test");

const load = () => import(pathToFileURL(join(__dirname, "../website/forest-lab/ml.mjs")).href);
test("CART separates XOR with depth, and a stump cannot", async () => {
  const ml = await load(), { train, test } = ml.makeDataset("xor", 0.02, 21);
  const deep = ml.trainModel("tree", train, 2, { depth: 4 });
  assert.ok(ml.accuracy(deep, train, 1) > 0.95 && ml.accuracy(deep, test, 1) > 0.9);
  const stump = ml.trainModel("tree", train, 2, { depth: 1 });
  assert.ok(ml.accuracy(stump, train, 1) < 0.75);
});

test("a perfectly separable pair of clusters is split by one threshold", async () => {
  const ml = await load();
  const pts = [-0.8, -0.6, -0.5, 0.5, 0.6, 0.9].map((x, i) => ({ x, y: i % 2 ? 0.4 : -0.4, c: x > 0 ? 1 : 0 }));
  const m = ml.trainModel("tree", pts, 2, { depth: 3 }), t = m.learners[0].tree;
  assert.equal(ml.accuracy(m, pts, 1), 1);
  assert.equal(t.f, 0);
  assert.ok(t.t > -0.5 && t.t < 0.5);
});

test("tree depth is capped and leaves are valid class distributions", async () => {
  const ml = await load(), { train } = ml.makeDataset("spiral", 0.05, 3);
  for (const depth of [1, 3, 6]) {
    const m = ml.trainModel("tree", train, 3, { depth });
    assert.ok(ml.treeDepth(m.learners[0].tree) <= depth);
    const probs = ml.scoresAt(m, 0.1, -0.2, 1);
    assert.ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  }
});

test("same seed gives the same forest; different seed gives a different one", async () => {
  const ml = await load(), { train } = ml.makeDataset("moons", 0.1, 5);
  const a = ml.trainModel("forest", train, 2, { depth: 5, learners: 15, seed: 9 });
  const b = ml.trainModel("forest", train, 2, { depth: 5, learners: 15, seed: 9 });
  const c = ml.trainModel("forest", train, 2, { depth: 5, learners: 15, seed: 10 });
  assert.deepEqual(JSON.stringify(a.learners), JSON.stringify(b.learners));
  assert.notEqual(JSON.stringify(a.learners), JSON.stringify(c.learners));
});

test("predictions do not change when called twice, and use only the first n learners", async () => {
  const ml = await load(), { train } = ml.makeDataset("blobs", 0.1, 2);
  const m = ml.trainModel("forest", train, 3, { depth: 4, learners: 20, seed: 1 });
  assert.deepEqual(ml.scoresAt(m, 0.3, 0.1, 7), ml.scoresAt(m, 0.3, 0.1, 7));
  const s = ml.scoresAt(m, 0.3, 0.1, 5);
  assert.ok(s.every(v => Math.abs(v * 5 - Math.round(v * 5)) < 1e-9), "hard votes from 5 trees come in fifths");
});

test("grid shares match point-by-point scores", async () => {
  const ml = await load(), { train } = ml.makeDataset("circles", 0.06, 4), G = 24;
  for (const kind of ["tree", "forest", "boost"]) {
    const m = ml.trainModel(kind, train, 2, { depth: kind === "boost" ? 2 : 4, learners: 12, seed: 2 });
    ml.computeGrid(m, G);
    const out = new Float32Array(G * G * 2), n = m.learners.length;
    ml.gridShares(m, n, out);
    for (const [i, j] of [[0, 0], [5, 17], [12, 12], [23, 3]]) {
      const want = ml.scoresAt(m, ml.cellCentre(i, G), ml.cellCentre(j, G), n), cell = j * G + i;
      for (let c = 0; c < 2; c++) assert.ok(Math.abs(out[cell * 2 + c] - want[c]) < 1e-5, `${kind} cell ${i},${j}`);
    }
  }
});

test("AdaBoost with stumps beats a single stump on rings; forest generalises on moons", async () => {
  const ml = await load();
  const rings = ml.makeDataset("circles", 0.06, 11);
  const stump = ml.trainModel("tree", rings.train, 2, { depth: 1 });
  const boost = ml.trainModel("boost", rings.train, 2, { depth: 1, learners: 40, seed: 1 });
  assert.ok(ml.accuracy(boost, rings.train, boost.learners.length) > ml.accuracy(stump, rings.train, 1) + 0.1);
  const moons = ml.makeDataset("moons", 0.12, 7);
  const forest = ml.trainModel("forest", moons.train, 2, { depth: 6, learners: 60, seed: 1 });
  assert.ok(ml.accuracy(forest, moons.test, 60) > 0.85);
});

test("held-out data is separate from training data and depth curve covers 1..12", async () => {
  const ml = await load(), d = ml.makeDataset("xor", 0.05, 6);
  const key = p => `${p.x.toFixed(6)},${p.y.toFixed(6)}`;
  const seen = new Set(d.train.map(key));
  assert.ok(d.test.every(p => !seen.has(key(p))));
  const rows = ml.depthCurve("tree", d.train, d.test, 2, { learners: 1, seed: 1 });
  assert.equal(rows.length, 12);
  assert.ok(rows[11].train >= rows[0].train);
});

test("voting ensemble: three different voters, majority wins, grid matches points", async () => {
  const ml = await load(), { train, test: held } = ml.makeDataset("moons", 0.12, 4), G = 20;
  const m = ml.trainModel("vote", train, 2, { depth: 3 });
  assert.deepEqual(m.learners.map(l => l.name), ["Decision tree", "kNN (k=5)", "Naive Bayes"]);
  const votes = [0, 1, 2].map(i => ml.vote(m, i, 0.1, 0.3)), top = votes.filter(v => v === 1).length >= 2 ? 1 : 0;
  assert.equal(ml.predictAt(m, 0.1, 0.3, 3), top);
  assert.ok(ml.accuracy(m, held, 3) > 0.8);
  ml.computeGrid(m, G);
  const out = new Float32Array(G * G * 2);
  ml.gridShares(m, 3, out);
  const want = ml.scoresAt(m, ml.cellCentre(7, G), ml.cellCentre(11, G), 3), cell = 11 * G + 7;
  assert.ok(Math.abs(out[cell * 2] - want[0]) < 1e-6);
});

test("real tabular data: stratified split, no overlap, scaling uses training rows only", async () => {
  const ml = await load();
  const rows = [];
  for (let i = 0; i < 90; i++) rows.push({ f: [i, (i * 7) % 13, i % 3], c: i % 3 });
  const d = ml.prepareReal(rows, 0, 1, 5);
  assert.equal(d.train.length + d.test.length, 90);
  assert.equal(d.K, 3);
  for (const c of [0, 1, 2]) assert.equal(d.test.filter(p => p.c === c).length, 6, "20% of each class is held out");
  const raw = p => p.raw.join(",");
  const seen = new Set(d.train.map(raw));
  assert.ok(d.test.every(p => !seen.has(raw(p))));
  const xs = d.train.map(p => p.x);
  assert.ok(Math.abs(Math.min(...xs) + 0.95) < 1e-9 && Math.abs(Math.max(...xs) - 0.95) < 1e-9);
  assert.ok(Math.abs(ml.unscale(0.95, d.ranges[0]) - d.ranges[0].max) < 1e-9);
  assert.deepEqual(JSON.stringify(ml.prepareReal(rows, 0, 1, 5)), JSON.stringify(d));
});

test("the bundled Wine CSV parses and a CART tree learns two of its features", async () => {
  const ml = await load(), fs = require("node:fs");
  const { names, rows } = ml.parseCsv(fs.readFileSync(join(__dirname, "../website/data/wine.csv"), "utf8"));
  assert.equal(names.length, 13);
  assert.equal(rows.length, 178);
  const d = ml.prepareReal(rows, names.indexOf("flavanoids"), names.indexOf("color_intensity"), 7);
  const tree = ml.trainModel("tree", d.train, 3, { depth: 4 });
  assert.ok(ml.accuracy(tree, d.test, 1) > 0.8);
  const base = Math.max(...[0, 1, 2].map(c => d.test.filter(p => p.c === c).length)) / d.test.length;
  assert.ok(ml.accuracy(tree, d.test, 1) > base + 0.2, "beats the majority baseline clearly");
});

test("layoutTree gives one box per node and counts leaves", async () => {
  const ml = await load(), { train } = ml.makeDataset("xor", 0.05, 8);
  const m = ml.trainModel("tree", train, 2, { depth: 3 }), t = m.learners[0].tree;
  const lay = ml.layoutTree(t);
  assert.equal(lay.nodes.length, ml.treeSize(t));
  assert.equal(lay.leaves, lay.nodes.filter(n => !n.n.l).length);
  assert.equal(lay.edges.length, lay.nodes.length - 1);
  assert.equal(t.n, train.length);
});
