// Builds each week page from code/weekN.py and runs the code in the browser with Pyodide (Python compiled to WebAssembly).
// Nothing is sent to a server: the data file and the code are executed locally in the visitor's browser.
const PYODIDE = window.PYODIDE_URL || "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
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

function loadScript(src) {
  return new Promise((ok, fail) => {
    const s = document.createElement("script"); s.src = src; s.onload = ok;
    s.onerror = () => fail(new Error("could not load " + src)); document.head.appendChild(s);
  });
}

function getPython(status) {
  if (!pyReady) pyReady = (async () => {
    status("Starting Python… (about 30 MB, only the first time)");
    await loadScript(PYODIDE + "pyodide.js");
    const py = await loadPyodide({ indexURL: PYODIDE });
    status("Loading numpy, pandas and matplotlib…");
    await py.loadPackage(["numpy", "pandas", "matplotlib"]);        // scipy and scikit-learn are loaded only for questions that import them
    status("Loading the Seattle weather data…");
    py.FS.mkdir("data");
    py.FS.writeFile("weather.py", await (await fetch("weather.py")).text());
    py.FS.writeFile("data/seattle_weather.csv", await (await fetch("data/seattle_weather.csv")).text());
    py.runPython(SETUP);
    return py;
  })().catch(e => { pyReady = null; throw e; });
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
  const [head, ...blocks] = text.split(/^# %% /m);
  const syllabus = head.split("\n").filter(l => l.startsWith("#:")).map(l => l.slice(2).trim()).join(" ");
  const questions = blocks.map(block => {
    const lines = block.split("\n"), [tag, title] = lines[0].split("|").map(s => s.trim());
    const rest = lines.slice(1);
    return {
      tag, title,
      description: rest.filter(l => l.startsWith("#:")).map(l => l.slice(2).trim()).join(" "),
      code: rest.filter(l => !l.startsWith("#:")).join("\n").trim() + "\n",
    };
  });
  return { syllabus, questions };
}

const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function buildQuestion(q, i) {
  const art = el("article", "q"); art.id = "q" + (i + 1);
  art.innerHTML = `
    <header class="q-head"><span class="q-num">${i + 1}</span>
      <div><span class="q-tag ${q.tag.startsWith("Practice") ? "practice" : ""}">${esc(q.tag)}</span><h3>${esc(q.title)}</h3></div></header>
    <p class="q-desc">${esc(q.description)}</p>
    <div class="runner">
      <div class="runner-body">
        <div class="code-col">
          <textarea spellcheck="false" aria-label="Python code for question ${i + 1}"></textarea>
          <div class="bar"><button class="run">▶ Run</button><button class="reset">Reset</button><span class="status"></span></div>
        </div>
        <div class="out-col">
          <p class="placeholder">Press <b>▶ Run</b> to see the output and plots here.</p>
          <pre class="out" hidden></pre>
          <div class="figs"></div>
        </div>
      </div>
    </div>`;
  const code = art.querySelector("textarea"); code.value = q.code;
  const run = art.querySelector(".run"), reset = art.querySelector(".reset");
  const out = art.querySelector(".out"), figs = art.querySelector(".figs"), status = art.querySelector(".status"), hint = art.querySelector(".placeholder");
  const fit = () => { code.style.height = "auto"; code.style.height = Math.min(code.scrollHeight + 4, 520) + "px"; };
  code.addEventListener("input", fit); requestAnimationFrame(fit);
  reset.onclick = () => { code.value = q.code; fit(); };
  run.onclick = async () => {
    run.disabled = true; out.textContent = ""; figs.innerHTML = ""; out.hidden = true; hint.hidden = true;
    const say = t => { status.textContent = t; };
    let release = () => {};
    try {
      if (pyReady) say("Waiting for Python to finish starting…");
      const py = await getPython(say);
      say("Loading the libraries this code needs…");
      await py.loadPackagesFromImports(code.value);
      const turn = busy; busy = new Promise(r => { release = r; });
      say("Waiting for the question that is running…"); await turn;
      say("Running…"); await new Promise(r => setTimeout(r, 0));   // let the browser paint the status first
      let text = ""; py.setStdout({ batched: s => { text += s + "\n"; } }); py.setStderr({ batched: s => { text += s + "\n"; } });
      const t0 = performance.now();
      try {
        py.runPython("_prepare(); plt.close('all')");
        await py.runPythonAsync(code.value, { globals: py.globals.get("dict")() });
        say("Done in " + ((performance.now() - t0) / 1000).toFixed(1) + " s");
      } catch (e) { text += String(e.message || e); say("Error – see the output"); }
      const plots = py.globals.get("_figs").toJs(); py.runPython("_figs.clear()");
      out.textContent = text; out.hidden = !text.trim();
      plots.forEach(svg => { const f = el("div", "fig", themedSvg(svg)); figs.appendChild(f); });
    } catch (e) { out.textContent = "Could not start Python in this browser: " + e.message; out.hidden = false; say(""); }
    finally { release(); }
    run.disabled = false;
  };
  return art;
}

async function buildPage(root) {
  const text = await (await fetch(root.dataset.src)).text();
  const { syllabus, questions } = parseQuestions(text);
  const box = document.getElementById("syllabus"), index = document.getElementById("qindex");
  if (box) box.innerHTML = `<b>Syllabus topics</b><p>${esc(syllabus)}</p>`;
  if (index) index.innerHTML = '<b>Questions on this page</b><ol>' + questions.map((q, i) =>
    `<li><a href="#q${i + 1}">${esc(q.title)}</a></li>`).join("") + "</ol>";
  questions.forEach((q, i) => root.appendChild(buildQuestion(q, i)));
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}

const root = document.getElementById("questions");
if (root) buildPage(root).catch(e => { root.textContent = "Could not load the questions: " + e.message; });
