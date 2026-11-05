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

test("Python bootstrap loads real weather; clustering and classifiers use built-in real datasets", async t => {
  const dom = new JSDOM(source("unit.html"), { url: "http://localhost/unit.html?u=1", runScripts: "outside-only" });
  const w = dom.window;
  t.after(() => w.close());
  w.matchMedia = () => ({ matches: false });
  w.requestIdleCallback = () => {};
  w.fetch = async path => ({ ok: true, text: async () => source(path) });
  const files = new Map();
  const py = {
    loadPackage: async () => {},
    FS: { mkdir() {}, writeFile: (path, contents) => files.set(path, contents) },
    runPython() {},
  };
  w.loadPyodide = async () => py;
  w.eval(source("output.js"));
  w.eval(source("sleep.js"));
  w.eval(source("run.js"));
  w.loadScript = async () => {};
  assert.equal(await w.getPython(), py);
  for (const unit of [1, 2]) {
    const paths = [...source(`code/week${unit}.py`).matchAll(/pd\.read_csv\("([^"]+)"\)/g)];
    assert.ok(paths.length > 0);
    for (const [, path] of paths) {
      assert.equal(files.get(path), source(path), `Unit ${unit} must read the loaded CSV`);
    }
  }
  assert.match(source("code/week3.py"), /load_wine\(as_frame=True\)/);
  assert.match(source("code/week5.py"), /load_breast_cancer\(as_frame=True\)/);
  for (const unit of [3, 5]) {
    assert.doesNotMatch(source(`code/week${unit}.py`), /read_csv|weather_regimes/);
  }
});

test("Unit 4 loads its signal CSV and recovers from a failed dataset download", async t => {
  const dom = new JSDOM(source("unit.html"), { url: "http://localhost/unit.html?u=4", runScripts: "outside-only" });
  const w = dom.window;
  t.after(() => w.close());
  w.matchMedia = () => ({ matches: false });
  w.requestIdleCallback = () => {};
  let fail = true, starts = 0;
  const files = new Map();
  w.fetch = async path => ({ ok: !(fail && path === "data/sleep_edf.csv"), text: async () => source(path) });
  w.loadPyodide = async () => {
    starts++;
    return { loadPackage: async () => {}, runPython() {},
      FS: { mkdir() {}, writeFile: (path, contents) => files.set(path, contents) } };
  };
  w.eval(source("output.js"));
  w.eval(source("sleep.js"));
  w.eval(source("run.js"));
  w.loadScript = async () => {};
  await assert.rejects(w.getPython(), /Could not load data\/sleep_edf.csv/);
  assert.equal(w.document.getElementById("py").dataset.state, "error");
  fail = false;
  await w.getPython();
  assert.equal(starts, 2);
  assert.equal(files.get("data/sleep_edf.csv"), source("data/sleep_edf.csv"));
  assert.equal(files.has("data/seattle_weather.csv"), false);
  assert.equal(w.document.getElementById("py").dataset.state, "ready");
});

async function page(t, unit = 1) {
  const dom = new JSDOM(source("unit.html"), { url: `http://localhost/unit.html?u=${unit}`, runScripts: "outside-only" });
  const w = dom.window;
  t.after(() => w.close());
  const timers = new Map();
  let timerId = 0;
  w.setTimeout = callback => { timers.set(++timerId, callback); return timerId; };
  w.clearTimeout = id => timers.delete(id);
  w.requestAnimationFrame = callback => queueMicrotask(callback);
  w.requestIdleCallback = () => {}; // Background preloading is tested separately from clicks.
  w.matchMedia = () => ({ matches: false });
  w.fetch = async path => ({ ok: true, text: async () => source(path) });
  w.eval(source("output.js"));
  w.eval(source("sleep.js"));
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
    if (!body) continue;                                // the Unit 4 explorer has no code panel
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

test("every practice can view its actual dataset before running Python", async t => {
  const datasets = {
    1: ["Seattle daily weather", 1461, 6, "2012-01-01"],
    2: ["Seattle daily weather", 1461, 6, "2012-01-01"],
    3: ["Wine", 178, 14, "14.23"],
    4: ["Sleep-EDF: sleep stages from EEG and eye movements", 200, 19, "SC400"],
    5: ["Wisconsin Diagnostic Breast Cancer", 569, 31, "17.99"],
  };
  for (const [unit, [name, rows, columns, firstValue]] of Object.entries(datasets)) {
    const p = await page(t, unit);
    for (const art of p.w.document.querySelectorAll(".q")) {
      const button = art.querySelector(".dataset-button");
      assert.ok(button, `Unit ${unit} ${art.id} has a Dataset button`);
      const panel = p.w.document.getElementById(button.getAttribute("aria-controls"));
      assert.equal(panel.hidden, true);
      button.click(); await p.settle();
      assert.equal(panel.hidden, false);
      assert.equal(button.getAttribute("aria-expanded"), "true");
      assert.equal(panel.querySelector("h2").textContent, name);
      assert.equal(panel.querySelectorAll("tbody tr").length, rows);
      assert.equal(panel.querySelectorAll("thead th").length, columns);
      if (Number(unit) === 4) {
        assert.match(panel.querySelector(".dataset-size").textContent, /21,948 rows/);
        assert.match(panel.querySelector(".dataset-size").textContent, /first 200 rows/);
        assert.match(panel.querySelector(".dataset-size").textContent, /extracted/);
        assert.match(panel.querySelector("a[href$='provenance.json']").textContent, /checksums/);
        assert.doesNotMatch(panel.querySelector("caption").textContent, /all original rows/);
      }
      assert.equal(panel.querySelector("tbody td").textContent, firstValue);
      assert.ok(panel.querySelector("a[download]").getAttribute("href").endsWith(".csv"));
      assert.equal(p.executed.length, 0);
      button.click();
      assert.equal(panel.hidden, true);
      assert.equal(button.getAttribute("aria-expanded"), "false");
    }
  }
});

test("dataset loading errors offer a retry and display cells as plain text", async t => {
  const p = await page(t), art = p.q(0).art;
  const button = art.querySelector(".dataset-button");
  assert.ok(button);
  let requests = 0;
  p.w.fetch = async () => {
    if (++requests === 1) return { ok: false };
    return { ok: true, text: async () => "date,weather\n2012-01-01,<img src=x onerror=alert(1)>\n" };
  };
  button.click(); await p.settle();
  assert.match(art.querySelector(".dataset-content").textContent, /Could not load/);
  button.click(); button.click(); await p.settle();
  assert.equal(requests, 2);
  assert.equal(art.querySelector(".dataset-content img"), null);
  assert.equal(art.querySelector("tbody td:last-child").textContent, "<img src=x onerror=alert(1)>");
});

test("dataset remains available while a practice runs and after its output appears", async t => {
  const p = await page(t), q = p.q(0), loading = deferred();
  p.w.getPython = () => loading.promise;
  const run = q.run.onclick(); await p.settle();
  const button = q.art.querySelector(".dataset-button");
  assert.ok(button);
  assert.equal(button.disabled, false);
  button.click(); await p.settle();
  assert.equal(q.art.querySelector(".dataset-panel").hidden, false);
  loading.resolve(p.py); await p.flush(); await run;
  assert.equal(q.art.querySelector(".dataset-panel").hidden, false);
  button.click();
  assert.equal(q.art.querySelector(".dataset-panel").hidden, true);
});

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

// ---- Unit 4 explorer: the page, DOM and handlers are real; only the Python boundary is modelled ----
const MODEL_END = "# 2. Leave one person out";
const SUBJECTS = ["SC400", "SC401", "SC402", "SC403", "SC404", "SC405", "SC406", "SC407"];
function fakeFold(subject) {
  const n = 240, expert = [], independent = [], decoded = [], posterior = [];
  for (let i = 0; i < n; i++) {
    const s = Math.floor(i / 60) % 5;
    expert.push(s); independent.push(i % 7 === 0 ? (s + 1) % 5 : s); decoded.push(i % 40 === 0 ? (s + 1) % 5 : s);
    posterior.push([0, 1, 2, 3, 4].map(k => (k === s ? 0.8 : 0.05)));
  }
  const matrix = hits => [0, 1, 2, 3, 4].map(r => [0, 1, 2, 3, 4].map(c => (r === c ? hits : 1)));
  const model = (accuracy, f1) => ({ accuracy, macro_f1: f1, stage_f1: [f1, f1, f1, f1, f1], confusion: matrix(accuracy * 40) });
  return JSON.stringify({
    subject, states: ["Wake", "N1", "N2", "N3", "REM"], features: Array.from({ length: 15 }, (_, k) => "f" + k),
    training_subjects: SUBJECTS.filter(s => s !== subject), training_segments: 1000, training_stage_counts: [500, 50, 250, 100, 100],
    segments: n, scored_segments: n, expert, majority: expert.map(() => 0), independent, decoded, posterior,
    eeg_delta: expert.map(v => 5 + v), eeg_beta: expert.map(v => 2 + v / 2),
    models: { majority: model(0.5, 0.2), independent: model(0.8, 0.6), decoded: model(0.9, 0.7) },
    transition: [0, 1, 2, 3, 4].map(r => [0, 1, 2, 3, 4].map(c => (r === c ? 0.9 : 0.025))),
    signature: [0, 1, 2, 3, 4].map(() => Array.from({ length: 15 }, (_, k) => k / 7 - 1)),
    log_likelihood: -1, row: {},
  });
}

async function explorer(t) {
  const p = await page(t, 4);
  p.w.HTMLCanvasElement.prototype.getContext = () => null;
  const calls = [];
  const namespace = { destroy() {}, get: name => name === "explore"
    ? Object.assign(subject => { calls.push(subject); return fakeFold(subject); }, { destroy() {} }) : undefined };
  p.py.loadPackagesFromImports = async () => {};
  p.py.runPython = (code, options) => code === "dict()" ? namespace : undefined;
  const art = p.w.document.querySelector(".q.sx");
  const card = s => art.querySelector(`[data-subject="${s}"]`);
  return { ...p, art, calls, card, namespace,
    state: s => card(s).querySelector(".sx-state").textContent,
    run: s => card(s).querySelector(".sx-run") };
}

test("Unit 4 opens on the explorer, with the full program kept as its own tab", async t => {
  const p = await explorer(t);
  const tabs = [...p.w.document.querySelectorAll(".tab")].map(b => b.textContent);
  assert.deepEqual(tabs, ["Explorer", "Full program"]);
  assert.equal(p.art.querySelectorAll(".sx-card").length, 8);
  assert.deepEqual([...p.art.querySelectorAll(".sx-card")].map(c => c.dataset.subject), SUBJECTS);
  assert.ok(p.w.document.querySelector("#q2 .run"), "the complete program still has its Run button");
  assert.match(p.w.document.querySelector("#q2 textarea").value, /def run_fold/);
});

test("the page's person list matches the people in the dataset", () => {
  const people = [...new Set(source("data/sleep_edf.csv").trim().split("\n").slice(1).map(line => line.split(",")[0]))];
  assert.deepEqual(people, SUBJECTS);
});

test("the explorer reuses the week4.py model code and fails loudly if its marker moves", async t => {
  const w = new JSDOM(source("unit.html"), { url: "http://localhost/unit.html?u=4", runScripts: "outside-only" }).window;
  t.after(() => w.close());
  w.matchMedia = () => ({ matches: false });
  w.fetch = async path => ({ ok: true, text: async () => source(path) });
  w.requestIdleCallback = () => {};
  w.eval(source("output.js")); w.eval(source("sleep.js")); w.eval(source("run.js"));
  const week4 = source("code/week4.py");
  const code = w.sleepModelCode(week4);
  for (const name of ["fit_hmm", "log_emissions", "viterbi", "forward_backward", "decode_record", "evaluate", "run_fold"]) {
    assert.match(code, new RegExp(`def ${name}\\(`), name);
  }
  assert.doesNotMatch(code, /for held_out in subjects/, "the all-people loop is not part of the reusable model code");
  assert.ok(week4.includes(MODEL_END));
  assert.throws(() => w.sleepModelCode(week4.replace(MODEL_END, "# 2. Evaluate")), /no longer has the line/);
  await new Promise(r => setImmediate(r));
});

test("Run trains one person live, locks every button, and shows that person's own result", async t => {
  const p = await explorer(t);
  p.run("SC403").click();
  assert.equal(p.run("SC400").disabled, true, "all buttons lock synchronously on the first click");
  assert.equal(p.art.querySelector(".sx-all").disabled, true);
  p.run("SC405").click();                               // ignored: a run is already accepted
  await p.flush();
  assert.deepEqual(p.calls, ["SC403"]);
  assert.match(p.state("SC403"), /HMM 90\.0%/);
  assert.equal(p.state("SC405"), "Not run yet");
  assert.match(p.art.querySelector(".sx-head h2").textContent, /^SC403/);
  const head = p.art.querySelector(".sx-head p").textContent;
  assert.doesNotMatch(head, /trained on[^(]*SC403/, "the tested person is not in the training list");
  assert.match(head, /SC400, SC401, SC402, SC404, SC405, SC406, SC407/);
  assert.equal(p.run("SC400").disabled, false);
  assert.equal(p.card("SC403").classList.contains("selected"), true);
  assert.equal(p.art.querySelectorAll(".sx-chart .plot svg").length > 0, true);
});

test("Run all 8 runs everyone in order and pools the confusion matrices", async t => {
  const p = await explorer(t);
  p.art.querySelector(".sx-all").click();
  await p.flush();
  assert.deepEqual(p.calls, SUBJECTS);
  assert.match(p.art.querySelector(".sx-people-table h3").textContent, /pooled over 8 of 8/);
  assert.equal(p.art.querySelectorAll(".sx-people-table tbody tr").length, 8);
  p.art.querySelector('tr[data-pick="SC401"]').click();
  assert.match(p.art.querySelector(".sx-head h2").textContent, /^SC401/);
});

test("clicking a finished card shows that person without running Python again", async t => {
  const p = await explorer(t);
  p.run("SC400").click(); await p.flush();
  p.run("SC401").click(); await p.flush();
  assert.match(p.art.querySelector(".sx-head h2").textContent, /^SC401/);
  p.card("SC400").click();
  assert.match(p.art.querySelector(".sx-head h2").textContent, /^SC400/);
  assert.deepEqual(p.calls, ["SC400", "SC401"]);
});

test("a failed run reports the error, unlocks the buttons and a retry succeeds", async t => {
  const p = await explorer(t);
  const explore = p.namespace.get;
  p.namespace.get = () => Object.assign(() => { throw new Error("PythonError: boom"); }, { destroy() {} });
  p.run("SC402").click(); await p.flush();
  assert.match(p.art.querySelector(".sx-error").textContent, /Could not run SC402\. .*boom/);
  assert.equal(p.run("SC402").disabled, false);
  assert.equal(p.state("SC402"), "Not run yet");
  p.namespace.get = explore;
  p.run("SC402").click(); await p.flush();
  assert.equal(p.art.querySelector(".sx-error"), null);
  assert.match(p.art.querySelector(".sx-head h2").textContent, /^SC402/);
});

test("explorer runs wait behind a running question on the shared interpreter", async t => {
  const p = await explorer(t), gate = deferred();
  const held = p.w.exclusive(() => gate.promise);        // another run holds the interpreter
  p.run("SC400").click(); await p.flush();
  assert.deepEqual(p.calls, [], "must not run while the interpreter is busy");
  assert.equal(p.run("SC401").disabled, true);
  gate.resolve(); await held; await p.flush();
  assert.deepEqual(p.calls, ["SC400"]);
});

test("hovering and arrow keys read the 30-second segment under the cursor", async t => {
  const p = await explorer(t);
  p.run("SC400").click(); await p.flush();
  const chart = p.art.querySelector(".sx-chart");
  chart.dispatchEvent(new p.w.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
  const read = p.art.querySelector(".sx-read").textContent;
  assert.match(read, /Expert/); assert.match(read, /Independent/); assert.match(read, /HMM/);
  const buttons = [...p.art.querySelectorAll(".sx-seg button")];
  buttons.find(b => b.textContent === "1 h").click();
  assert.match(p.art.querySelector(".sx-window").textContent, /independent wrong on/);
});
