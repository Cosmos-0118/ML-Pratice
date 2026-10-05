// Unit page: shows one practice question at a time from code/weekN.py and runs it in the browser with Pyodide
// (Python compiled to WebAssembly). Nothing is sent to a server. The output appears in the same box as the code.
const PYODIDE = window.PYODIDE_URL || "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const UNITS = {
  1: "Introduction",
  2: "Linear models for regression",
  3: "Mixture models and EM",
  4: "Hidden Markov models",
  5: "Combining models",
};
let pyReady = null;
let busy = Promise.resolve();     // questions run one at a time, so their output and plots never get mixed up

// Plots are drawn as SVG (sharp at any size). Text and axis colours are swapped for the page's colour variables,
// so the plots follow light / dark mode, even after the theme is toggled.
const SETUP = `
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt, io, weather
_figs = []
def _prepare():
    weather.use_style()
    plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": "#010101", "axes.labelcolor": "#010101", "axes.edgecolor": "#020202",
        "xtick.color": "#020202", "ytick.color": "#020202", "grid.color": "#030303", "svg.fonttype": "path"})
def _show(*a, **k):
    for n in plt.get_fignums():
        b = io.StringIO(); plt.figure(n).savefig(b, format="svg", metadata={"Date": None}); _figs.append(b.getvalue())
    plt.close("all")
plt.show = _show
_prepare()
`;

// ---- Python status: the dot in the top bar, plus the status line of a question that is waiting ----
const pyDot = document.getElementById("py");
const listeners = new Set();
function pyStatus(state, text) {
  if (pyDot) { pyDot.dataset.state = state; pyDot.title = text; pyDot.querySelector(".py-text").textContent = state === "ready" ? "Python ready" : state === "error" ? "Python failed" : "Loading Python…"; }
  listeners.forEach(f => f(text));
}

function loadScript(src) {
  return new Promise((ok, fail) => {
    const s = document.createElement("script"); s.src = src; s.onload = ok;
    s.onerror = () => fail(new Error("could not load " + src)); document.head.appendChild(s);
  });
}

function getPython() {
  if (!pyReady) pyReady = (async () => {
    pyStatus("loading", "Starting Python… (about 30 MB, only the first time)");
    await loadScript(PYODIDE + "pyodide.js");
    const py = await loadPyodide({ indexURL: PYODIDE });
    pyStatus("loading", "Loading numpy, pandas and matplotlib…");
    await py.loadPackage(["numpy", "pandas", "matplotlib"]);        // scipy and scikit-learn are loaded only for questions that import them
    pyStatus("loading", "Loading the Seattle weather data…");
    py.FS.mkdir("data");
    py.FS.writeFile("weather.py", await (await fetch("weather.py")).text());
    py.FS.writeFile("data/seattle_weather.csv", await (await fetch("data/seattle_weather.csv")).text());
    py.runPython(SETUP);
    pyStatus("ready", "Python is ready");
    return py;
  })().catch(e => { pyReady = null; pyStatus("error", "Could not start Python: " + e.message); throw e; });
  return pyReady;
}

// ---- turn a plot from matplotlib into inline SVG that follows the page theme ----
function themedSvg(svg) {
  return svg
    .replace(/<\?xml[\s\S]*?\?>|<!DOCTYPE[\s\S]*?>|<!--[\s\S]*?-->/g, "")
    .replace(/<svg([^>]*?)\swidth="[^"]*"/, "<svg$1").replace(/<svg([^>]*?)\sheight="[^"]*"/, "<svg$1")
    .replace(/#010101/gi, "var(--plot-text)").replace(/#020202/gi, "var(--plot-axis)").replace(/#030303/gi, "var(--plot-grid)");
}

// ---- parse code/weekN.py: "# %% Tag | Title", "#: description" lines, then the code ----
function parseQuestions(text) {
  return text.split(/^# %% /m).slice(1).map(block => {
    const lines = block.split("\n"), [tag, title] = lines[0].split("|").map(s => s.trim());
    const rest = lines.slice(1);
    return {
      tag, title,
      description: rest.filter(l => l.startsWith("#:")).map(l => l.slice(2).trim()).join(" "),
      code: rest.filter(l => !l.startsWith("#:")).join("\n").trim() + "\n",
    };
  });
}

// ---- a small Python syntax highlighter for the editor ----
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const KEYWORDS = new Set("and as assert break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" "));
const BUILTINS = new Set("print range len zip enumerate min max sum abs round sorted list dict set tuple int float str bool isinstance map any all open".split(" "));
const TOKEN = /(#[^\n]*)|((?:\b[rRbBfFuU]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(\b\d+\.?\d*(?:[eE][-+]?\d+)?\b|\.\d+\b)|([A-Za-z_]\w*)/g;
function highlight(code) {
  let html = "", last = 0, m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(code))) {
    html += esc(code.slice(last, m.index)); last = TOKEN.lastIndex;
    const [t, com, str, num, word] = m;
    let cls = com ? "com" : str ? "str" : num ? "num" : "";
    if (word) cls = KEYWORDS.has(word) ? "kw" : BUILTINS.has(word) ? "bi" : code[TOKEN.lastIndex] === "(" ? "fn" : "";
    html += cls ? `<span class="hl-${cls}">${esc(t)}</span>` : esc(t);
  }
  return html + esc(code.slice(last)) + "\n";
}

const ICON_PLAY = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.8v10.4a.8.8 0 0 0 1.2.7l8.4-5.2a.8.8 0 0 0 0-1.4L5.2 2.1a.8.8 0 0 0-1.2.7z"/></svg>';
const ICON_CODE = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 4 1.5 8l4 4M10.5 4l4 4-4 4"/></svg>';
const nextFrame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
const wait = ms => new Promise(r => setTimeout(r, ms));

// Smoothly change the height of the box while its content is swapped.
async function morph(body, swap) {
  const from = body.offsetHeight;
  swap();
  const to = body.scrollHeight;
  if (Math.abs(to - from) < 2 || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  body.style.height = from + "px"; body.classList.add("sizing");
  body.offsetHeight;                                            // apply the start height before animating
  body.style.height = to + "px";
  await wait(480);
  body.classList.remove("sizing"); body.style.height = "";
}

// One big picture at a time; switch with the arrow buttons, the ← → keys or a swipe.
const ARROW = d => `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${d}"/></svg>`;
function buildGallery(box, plots) {
  box.innerHTML = ""; box.go = null;
  if (!plots.length) return;
  const many = plots.length > 1;
  box.innerHTML = `
    <div class="slides">${plots.map((svg, i) => `<figure class="slide" aria-label="Plot ${i + 1} of ${plots.length}">${themedSvg(svg)}</figure>`).join("")}</div>
    ${many ? `<button class="gbtn prev" type="button" aria-label="Previous plot">${ARROW("M10 3 5 8l5 5")}</button>
    <button class="gbtn next" type="button" aria-label="Next plot">${ARROW("M6 3l5 5-5 5")}</button>
    <div class="gbar"><span class="gdots">${plots.map((_, i) => `<button type="button" aria-label="Plot ${i + 1}"></button>`).join("")}</span>
      <span class="gcount"></span><span class="ghint">← → keys</span></div>` : ""}`;
  const slides = [...box.querySelectorAll(".slide")], dots = [...box.querySelectorAll(".gdots button")], count = box.querySelector(".gcount");
  let at = 0;
  const show = (n, dir = 1) => {
    at = (n + slides.length) % slides.length;
    slides.forEach((s, i) => { s.classList.toggle("on", i === at); if (i !== at) s.style.setProperty("--from", (dir > 0 ? -1 : 1) * 28 + "px"); });
    slides[at].style.setProperty("--from", dir * 28 + "px");
    dots.forEach((d, i) => d.setAttribute("aria-current", i === at));
    if (count) count.textContent = `${at + 1} / ${slides.length}`;
  };
  show(0);
  if (!many) return;
  box.go = d => show(at + d, d);
  box.querySelector(".prev").onclick = () => box.go(-1);
  box.querySelector(".next").onclick = () => box.go(1);
  dots.forEach((d, i) => d.onclick = () => show(i, i >= at ? 1 : -1));
  let x0 = null;
  box.addEventListener("touchstart", e => { x0 = e.touches[0].clientX; }, { passive: true });
  box.addEventListener("touchend", e => {
    const dx = e.changedTouches[0].clientX - x0;
    if (x0 !== null && Math.abs(dx) > 40) box.go(dx < 0 ? 1 : -1);
    x0 = null;
  });
}
document.addEventListener("keydown", e => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  if (e.target.closest("textarea, input, select, [contenteditable]") || e.altKey || e.ctrlKey || e.metaKey) return;
  const box = document.querySelector(".q:not([hidden]) .output:not([hidden]) .figs");
  if (box && box.go) { e.preventDefault(); box.go(e.key === "ArrowRight" ? 1 : -1); }
});

function buildQuestion(q, i, unit) {
  const art = document.createElement("article");
  art.className = "q"; art.id = "q" + (i + 1); art.hidden = true;
  art.setAttribute("role", "tabpanel");
  art.innerHTML = `
    <p class="q-tag">${esc(q.tag)}</p>
    <h1 class="q-title">${esc(q.title)}</h1>
    <p class="q-desc">${esc(q.description)}</p>
    <div class="panel">
      <div class="halo" aria-hidden="true"></div><div class="fx" aria-hidden="true"></div>
      <div class="panel-inner">
        <div class="panel-bar">
          <span class="wdots" aria-hidden="true"><i></i><i></i><i></i></span>
          <span class="fname">unit${unit}_${q.tag.toLowerCase().replace(/\s+/g, "")}.py</span>
          <span class="spacer"></span>
          <span class="done" hidden></span>
          <button class="btn reset" type="button" title="Undo your edits">Reset</button>
          <button class="btn show-code" type="button" hidden>${ICON_CODE} Code</button>
          <button class="btn primary run" type="button" title="Run (Ctrl / ⌘ + Enter)">${ICON_PLAY} Run</button>
        </div>
        <div class="panel-body">
          <div class="editor"><pre aria-hidden="true"></pre><textarea spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="Python code"></textarea></div>
          <div class="output" hidden aria-live="polite"><pre class="out"></pre><div class="figs"></div></div>
          <div class="veil" aria-hidden="true"><span class="sweep"></span><span class="veil-status">Running…</span></div>
        </div>
      </div>
    </div>`;
  const $ = s => art.querySelector(s);
  const panel = $(".panel"), body = $(".panel-body"), editor = $(".editor"), pre = $(".editor pre"), code = $("textarea");
  const output = $(".output"), out = $(".out"), figs = $(".figs"), veilStatus = $(".veil-status");
  const run = $(".run"), reset = $(".reset"), showCode = $(".show-code"), done = $(".done"), fname = $(".fname");
  const codeName = fname.textContent;

  const paint = () => { pre.innerHTML = highlight(code.value); };
  code.value = q.code; paint();
  code.addEventListener("input", paint);
  code.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run.click(); }
    else if (e.key === "Tab" && !e.shiftKey) { e.preventDefault(); document.execCommand("insertText", false, "    "); }
  });
  reset.onclick = () => { code.value = q.code; paint(); };

  const setMode = mode => {                                       // "code" or "output"
    const showOut = mode === "output";
    editor.hidden = showOut; output.hidden = !showOut;
    reset.hidden = showOut; showCode.hidden = !showOut;
    run.innerHTML = ICON_PLAY + (showOut ? " Run again" : " Run");
    fname.textContent = showOut ? "output" : codeName;
    if (!showOut) done.hidden = true;
  };
  showCode.onclick = async () => {
    editor.classList.remove("enter"); void editor.offsetWidth; editor.classList.add("enter");
    await morph(body, () => setMode("code"));
  };

  run.onclick = async () => {
    if (panel.classList.contains("running")) return;
    // 1. start the animation over the code
    if (!output.hidden) await morph(body, () => setMode("code"));
    panel.style.setProperty("--fx-size", Math.ceil(Math.hypot(panel.offsetWidth, panel.offsetHeight) + 60) + "px");
    panel.classList.add("running"); run.disabled = true; reset.disabled = true; done.hidden = true;
    const say = t => { veilStatus.textContent = t; };
    listeners.add(say);
    let release = () => {}, text = "", plots = [], failed = false, seconds = 0;
    try {
      say(pyReady ? "Starting Python…" : "Starting Python… (about 30 MB, only the first time)");
      const py = await getPython();
      say("Loading the libraries this code needs…");
      await py.loadPackagesFromImports(code.value);
      const turn = busy; busy = new Promise(r => { release = r; });
      say("Waiting for the other question to finish…"); await turn;
      say("Running…");
      await nextFrame(); await wait(250);                           // let the animation start before Python takes the main thread
      py.setStdout({ batched: s => { text += s + "\n"; } }); py.setStderr({ batched: s => { text += s + "\n"; } });
      const t0 = performance.now();
      try {
        py.runPython("_prepare(); plt.close('all')");
        await py.runPythonAsync(code.value, { globals: py.globals.get("dict")() });
      } catch (e) { failed = true; text += String(e.message || e); }
      seconds = (performance.now() - t0) / 1000;
      plots = py.globals.get("_figs").toJs(); py.runPython("_figs.clear()");
    } catch (e) { failed = true; text = "Could not start Python in this browser: " + e.message; }
    finally { release(); listeners.delete(say); }

    // 2. the code turns into its output, in the same place
    out.textContent = text.replace(/\n+$/, ""); out.classList.toggle("err", failed);
    buildGallery(figs, plots);
    say(failed ? "Error" : "Done");
    panel.classList.remove("running");
    output.classList.remove("enter"); void output.offsetWidth; output.classList.add("enter");
    await morph(body, () => setMode("output"));
    done.textContent = failed ? "Error" : `Done in ${seconds.toFixed(1)} s`; done.hidden = false;
    run.disabled = false; reset.disabled = false;
  };
  return art;
}

async function buildPage() {
  const unit = Math.min(5, Math.max(1, parseInt(new URLSearchParams(location.search).get("u"), 10) || 1));
  document.title = `Unit ${unit} – ${UNITS[unit]} | ML Practice`;
  document.getElementById("crumb").innerHTML = `<b>Unit ${unit}</b> · ${esc(UNITS[unit])}`;
  const root = document.getElementById("questions"), tabs = document.getElementById("tabs"), nav = document.getElementById("nextnav");
  const questions = parseQuestions(await (await fetch(`code/week${unit}.py`, { cache: "no-cache" })).text());
  const arts = questions.map((q, i) => buildQuestion(q, i, unit));
  root.replaceChildren(...arts);

  const buttons = questions.map((q, i) => {
    const b = document.createElement("button");
    b.className = "tab"; b.type = "button"; b.textContent = q.tag; b.setAttribute("role", "tab");
    b.onclick = () => show(i, true);
    return b;
  });
  tabs.replaceChildren(...buttons);
  tabs.hidden = questions.length < 2;

  function show(i, push) {
    arts.forEach((a, j) => { a.hidden = j !== i; buttons[j].setAttribute("aria-selected", j === i); });
    if (push) history.replaceState(null, "", `?u=${unit}#q${i + 1}`);
    const link = (href, label, title, cls) => `<a class="${cls}" href="${href}">
      <span class="ico">${ARROW(cls === "fwd" ? "M6 3l5 5-5 5" : "M10 3 5 8l5 5")}</span>
      <span class="txt"><small>${label}</small><strong>${esc(title)}</strong></span></a>`;
    let html = i > 0 ? link(`#q${i}`, "Previous", questions[i - 1].tag, "prv")
      : unit > 1 ? link(`unit.html?u=${unit - 1}`, "Previous unit", `Unit ${unit - 1} · ${UNITS[unit - 1]}`, "prv") : "";
    if (i < questions.length - 1) html += link(`#q${i + 2}`, "Next", questions[i + 1].tag, "fwd");
    else if (unit < 5) html += link(`unit.html?u=${unit + 1}`, "Next unit", `Unit ${unit + 1} · ${UNITS[unit + 1]}`, "fwd");
    else html += link("index.html", "Finished", "Back to all units", "fwd");
    nav.innerHTML = html;
    nav.querySelectorAll('a[href^="#q"]').forEach(a => a.onclick = e => {
      e.preventDefault(); show(parseInt(a.getAttribute("href").slice(2), 10) - 1, true); scrollTo({ top: 0, behavior: "smooth" });
    });
  }
  const start = Math.max(0, Math.min(questions.length - 1, (parseInt(location.hash.slice(2), 10) || 1) - 1));
  show(start, false);

  // Start Python in the background, so the first Run is quicker.
  (window.requestIdleCallback || (f => setTimeout(f, 800)))(() => getPython().catch(() => {}), { timeout: 2000 });
}

buildPage().catch(e => { document.getElementById("questions").textContent = "Could not load the questions: " + e.message; });
