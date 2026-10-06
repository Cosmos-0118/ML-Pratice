const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { JSDOM } = require("jsdom");

const root = join(__dirname, "../website");
const source = name => readFileSync(join(root, name), "utf8");
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
};

async function page(t) {
  const dom = new JSDOM(source("unit.html"), { url: "http://localhost/unit.html?u=1", runScripts: "outside-only" });
  const w = dom.window;
  t.after(() => w.close());
  const timers = new Map();
  let timerId = 0;
  w.setTimeout = callback => { timers.set(++timerId, callback); return timerId; };
  w.clearTimeout = id => timers.delete(id);
  w.requestAnimationFrame = callback => queueMicrotask(callback);
  w.requestIdleCallback = () => {}; // Background preloading is tested separately from clicks.
  w.matchMedia = () => ({ matches: false });
  w.fetch = async () => ({ ok: true, text: async () => source("code/week1.py") });
  w.eval(source("output.js"));
  w.eval(source("run.js"));
  const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
  const flush = async () => {
    for (let i = 0; i < 15; i++) {
      await settle();
      const current = [...timers.values()]; timers.clear();
      current.forEach(callback => callback());
    }
    await settle();
  };
  await settle();
  const articles = [...w.document.querySelectorAll(".q")];
  for (const art of articles) {
    const body = art.querySelector(".panel-body");
    const height = () => art.querySelector(".editor").hidden ? 400 : 200;
    Object.defineProperties(body, { offsetHeight: { get: height }, scrollHeight: { get: height } });
  }
  let stdout;
  const executed = [];
  // Model only the Python boundary; the real page, DOM, handlers and formatter run above.
  const py = {
    loadPackagesFromImports: async () => {},
    setStdout: stream => { stdout = stream.batched; }, setStderr: () => {},
    runPython: code => code === "dict()" ? { destroy() {} } : undefined,
    runPythonAsync: async code => { executed.push(code); stdout(code); },
    globals: { get: name => name === "dict" ? () => ({ destroy() {} }) : { toJs: () => [], destroy() {} } },
  };
  w.getPython = async () => py;
  const q = index => {
    const art = articles[index];
    return { art, run: art.querySelector(".run"), reset: art.querySelector(".reset"),
      code: art.querySelector("textarea"), output: art.querySelector(".output"), out: art.querySelector(".out"),
      panel: art.querySelector(".panel"), done: art.querySelector(".done") };
  };
  return { w, py, q, executed, settle, flush };
}

test("a rerun locks immediately and rapid clicks execute only once", async t => {
  const p = await page(t), q = p.q(0);
  q.output.hidden = false;
  const first = q.run.onclick();
  const lockedImmediately = q.run.disabled;
  const second = q.run.onclick();
  await p.flush(); await Promise.all([first, second]);
  assert.equal(lockedImmediately, true, "accepting a click must lock before any animation await");
  assert.equal(p.executed.length, 1, "one rerun must produce one execution");
  assert.equal(q.run.disabled, false);
  assert.match(q.done.textContent, /Done/);
});

test("pending work is visibly busy instead of showing an enabled-looking Run label", async t => {
  const p = await page(t), q = p.q(0), loading = deferred();
  p.w.getPython = () => loading.promise;
  const run = q.run.onclick(); await p.settle();
  const label = q.run.textContent;
  loading.resolve(p.py); await p.flush(); await run;
  assert.match(label, /Loading|Starting|Running|Waiting/);
  assert.equal(q.run.disabled, false);
});

test("the accepted click runs its code even if the editor changes during loading", async t => {
  const p = await page(t), q = p.q(0), loading = deferred();
  q.code.value = 'print("accepted")';
  p.w.getPython = () => loading.promise;
  const run = q.run.onclick(); await p.settle();
  q.code.value = 'print("later edit")';
  loading.resolve(p.py); await p.flush(); await run;
  assert.equal(q.out.textContent, 'print("accepted")');
});

test("a display failure restores Run and allows a subsequent successful retry", async t => {
  const p = await page(t), q = p.q(0), format = p.w.formatOutput;
  p.w.formatOutput = () => { throw new Error("display failed"); };
  const first = q.run.onclick(); first.catch(() => {});
  await p.flush(); await first.catch(() => {});
  assert.equal(q.run.disabled, false);
  assert.equal(q.reset.disabled, false);
  assert.equal(q.panel.classList.contains("running"), false);
  assert.match(q.out.textContent, /display failed/);
  p.w.formatOutput = format;
  const retry = q.run.onclick(); await p.flush(); await retry;
  assert.equal(q.out.classList.contains("err"), false);
  assert.match(q.done.textContent, /Done/);
});

test("question runs serialize package loading along with execution", async t => {
  const p = await page(t), one = p.q(0), two = p.q(1), loading = deferred();
  let activeLoads = 0;
  p.py.loadPackagesFromImports = async () => {
    activeLoads++;
    if (activeLoads === 1) await loading.promise;
  };
  const first = one.run.onclick(); await p.settle();
  const second = two.run.onclick(); await p.settle();
  const loadsBeforeRelease = activeLoads;
  loading.resolve(); await p.flush(); await Promise.all([first, second]);
  assert.equal(loadsBeforeRelease, 1, "the shared interpreter must not load packages concurrently");
  assert.equal(p.executed.length, 2);
  assert.equal(one.run.disabled, false);
  assert.equal(two.run.disabled, false);
});

test("a Python or package failure does not strand later question runs", async t => {
  const p = await page(t), one = p.q(0), two = p.q(1);
  p.py.loadPackagesFromImports = async () => { throw new Error("package unavailable"); };
  const first = one.run.onclick(); await p.flush(); await first;
  assert.equal(one.run.disabled, false);
  assert.match(one.out.textContent, /package unavailable/);
  p.py.loadPackagesFromImports = async () => {};
  const second = two.run.onclick(); await p.flush(); await second;
  assert.match(two.done.textContent, /Done/);
});

test("a Python exception preserves printed output and the next run recovers", async t => {
  const p = await page(t), q = p.q(0), execute = p.py.runPythonAsync;
  p.py.runPythonAsync = async code => {
    await execute(code);
    throw new Error("ValueError: test exception");
  };
  const first = q.run.onclick(); await p.flush(); await first;
  assert.ok(q.out.textContent.startsWith(q.code.value));
  assert.match(q.out.textContent, /ValueError: test exception/);
  assert.equal(q.run.disabled, false);
  p.py.runPythonAsync = execute;
  const retry = q.run.onclick(); await p.flush(); await retry;
  assert.equal(q.out.classList.contains("err"), false);
  assert.match(q.done.textContent, /Done/);
});

test("an early failure of a queued question cannot let its retry bypass an active run", async t => {
  const p = await page(t), one = p.q(0), two = p.q(1), loading = deferred();
  let loads = 0;
  p.py.loadPackagesFromImports = async () => { if (++loads === 1) await loading.promise; };
  const first = one.run.onclick(); await p.settle();
  const style = two.panel.style, setProperty = style.setProperty.bind(style);
  style.setProperty = () => { throw new Error("animation failed"); };
  const failed = two.run.onclick(); await p.settle(); await failed;
  assert.equal(two.run.disabled, false);
  style.setProperty = setProperty;
  const retry = two.run.onclick(); await p.flush();
  const loadsBeforeRelease = loads;
  loading.resolve(); await p.flush(); await Promise.all([first, retry]);
  assert.equal(loadsBeforeRelease, 1, "a failed ticket must still wait for the preceding ticket");
  assert.equal(p.executed.length, 2);
});
