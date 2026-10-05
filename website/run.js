// Runs the Python snippets in the browser with Pyodide (Python compiled to WebAssembly).
// Nothing is sent to a server: the data file and the code are executed locally in the visitor's browser.
const PYODIDE = window.PYODIDE_URL || "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let pyReady = null;

const SETUP = `
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt, io, base64
_figs = []
def _show(*a, **k):
    for n in plt.get_fignums():
        b = io.BytesIO(); plt.figure(n).savefig(b, format="png", dpi=110, bbox_inches="tight")
        _figs.append(base64.b64encode(b.getvalue()).decode())
    plt.close("all")
plt.show = _show
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
    status("Loading numpy, pandas, scikit-learn, matplotlib…");
    await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]);
    status("Loading the Seattle weather data…");
    py.FS.mkdir("data");
    py.FS.writeFile("weather.py", await (await fetch("weather.py")).text());
    py.FS.writeFile("data/seattle_weather.csv", await (await fetch("data/seattle_weather.csv")).text());
    py.runPython(SETUP);
    return py;
  })().catch(e => { pyReady = null; throw e; });
  return pyReady;
}

document.querySelectorAll(".runner").forEach(box => {
  const code = box.querySelector("textarea");
  let original = "";
  const run = box.querySelector(".run"), reset = box.querySelector(".reset");
  const out = box.querySelector(".out"), figs = box.querySelector(".figs"), status = box.querySelector(".status");
  const fit = () => { code.style.height = "auto"; code.style.height = Math.min(code.scrollHeight + 4, 560) + "px"; };
  code.addEventListener("input", fit);
  // The code lives in code/weekN.py, so the page and the file can never drift apart.
  fetch(code.dataset.src).then(r => r.text()).then(t => { original = t; code.value = t; fit(); })
    .catch(() => { code.value = "# Could not load " + code.dataset.src; });
  reset.onclick = () => { code.value = original; fit(); };
  run.onclick = async () => {
    run.disabled = true; out.textContent = ""; figs.innerHTML = ""; out.hidden = true;
    const say = t => { status.textContent = t; };
    try {
      const py = await getPython(say);
      say("Running…"); await new Promise(r => setTimeout(r, 30));
      let text = ""; py.setStdout({ batched: s => { text += s + "\n"; } }); py.setStderr({ batched: s => { text += s + "\n"; } });
      const t0 = performance.now();
      try {
        await py.runPythonAsync(code.value, { globals: py.globals.get("dict")() });
        say("Done in " + ((performance.now() - t0) / 1000).toFixed(1) + " s");
      } catch (e) { text += String(e.message || e); say("Error – see the output below"); }
      const imgs = py.globals.get("_figs").toJs(); py.globals.get("_figs").destroy(); py.runPython("_figs.clear()");
      out.textContent = text; out.hidden = !text.trim();
      imgs.forEach(b64 => { const i = new Image(); i.src = "data:image/png;base64," + b64; figs.appendChild(i); });
    } catch (e) { out.textContent = "Could not start Python in this browser: " + e.message; out.hidden = false; say(""); }
    run.disabled = false;
  };
});
