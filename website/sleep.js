// Unit 4 explorer. Pick a person: Python trains the HMM on the other seven people, decodes that person's whole
// recording and this file draws the result. The model is the code in code/week4.py (everything above its
// "# 2. Leave one person out" line) plus the small JSON glue in code/sleep_bridge.py; nothing is modelled here.
// Loaded before run.js, which calls buildSleepExplorer(q, i, unit, kit) and hands over what it needs in `kit`
// (getPython, exclusive, nextFrame, wait, esc, highlight, DATASETS, ICON_PLAY, datasetPanelMarkup, wireDataset).
let kit = null;
const SLEEP_SUBJECTS = ["SC400", "SC401", "SC402", "SC403", "SC404", "SC405", "SC406", "SC407"];
const SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"];
const SLEEP_MODEL_END = "# 2. Leave one person out";
const SLEEP_MODELS = [
  { key: "majority", name: "Training-majority baseline", note: "always guesses the most common stage" },
  { key: "independent", name: "Independent Gaussian", note: "each 30 s on its own" },
  { key: "decoded", name: "HMM · Viterbi", note: "the whole night together" },
];
const SLEEP_WINDOWS = [["Full night", 0], ["6 h", 6], ["3 h", 3], ["1 h", 1]];
const SEGMENTS_PER_HOUR = 120;

const pct = v => (v * 100).toFixed(1) + "%";
const clock = h => { const m = Math.round(h * 60); return Math.floor(m / 60) + ":" + String(m % 60).padStart(2, "0"); };
const stageDot = s => `<i class="sx-dot ${s < 0 ? "su" : "s" + s}"></i>`;
const stageName = s => s < 0 ? "Unscored" : SLEEP_STAGES[s];

// accuracy and macro F1 from a 5×5 confusion matrix (rows: expert, columns: prediction), as evaluate() in week4.py
function scoreOf(matrix) {
  let total = 0, correct = 0, f1 = 0;
  const stageF1 = matrix.map((row, k) => {
    const column = matrix.reduce((sum, r) => sum + r[k], 0), rowSum = row.reduce((a, b) => a + b, 0);
    total += rowSum; correct += row[k];
    return rowSum + column > 0 ? 2 * row[k] / (rowSum + column) : 0;
  });
  stageF1.forEach(v => { f1 += v / matrix.length; });
  return { accuracy: total ? correct / total : 0, macroF1: f1, stageF1 };
}
const addMatrices = (a, b) => a.map((row, i) => row.map((v, j) => v + b[i][j]));
const switches = seq => { let n = 0; for (let i = 1; i < seq.length; i++) if (seq[i] !== seq[i - 1]) n++; return n; };

// the model code that runs on every click, shown read-only under the page
function sleepModelCode(week4) {
  const block = week4.split(/^# %% /m)[1] || "";
  const code = block.split("\n").slice(1).filter(line => !line.startsWith("#:")).join("\n"), cut = code.indexOf(SLEEP_MODEL_END);
  if (cut < 0) throw new Error("code/week4.py no longer has the line “" + SLEEP_MODEL_END + "”");
  return code.slice(0, cut).trimEnd() + "\n";
}

let sleepNamespace = null;          // Python globals holding week4's model code and explore(); built once, kept for every click
let sleepSource = null;
async function loadSleepModel(py) {
  if (sleepNamespace) return sleepNamespace;
  const [week4, bridge] = await Promise.all(["code/week4.py", "code/sleep_bridge.py"].map(async path => {
    const response = await fetch(path, { cache: "no-cache" });
    if (!response.ok) throw new Error("Could not load " + path);
    return response.text();
  }));
  const model = sleepModelCode(week4);
  await py.loadPackagesFromImports(model);
  const namespace = py.runPython("dict()");
  try {
    py.runPython(model, { globals: namespace });
    py.runPython(bridge, { globals: namespace });
  } catch (e) { namespace.destroy(); throw e; }
  sleepSource = model;
  return (sleepNamespace = namespace);
}

// Train on the other seven people and decode `subject`; shares the interpreter with the other pages' runs.
async function runSleepFold(subject) {
  const py = await kit.getPython();
  return kit.exclusive(async () => {
    const namespace = await loadSleepModel(py);
    await kit.nextFrame(); await kit.wait(60);                  // let the "running" state paint before Python takes the main thread
    const explore = namespace.get("explore");
    try { return JSON.parse(explore(subject)); } finally { explore.destroy(); }
  });
}

// ---- small drawing helpers ----
function eachRun(start, end, key, draw) {
  let from = start;
  for (let i = start + 1; i <= end; i++) {
    if (i === end || key(i) !== key(from)) { draw(from, i, key(from)); from = i; }
  }
}

function stripSvg(seq, expert, start, end, mistakes) {
  const n = end - start, h = 42;
  let rects = "";
  eachRun(start, end, i => seq[i], (a, b, s) =>
    rects += `<rect class="${s < 0 ? "su" : "s" + s}" x="${a - start}" y="0" width="${b - a}" height="${h}"/>`);
  if (mistakes && expert) {
    eachRun(start, end, i => expert[i] >= 0 && seq[i] !== expert[i], (a, b, wrong) => {
      if (wrong) rects += `<rect class="miss" x="${a - start}" y="34" width="${b - a}" height="8"/>`;
    });
  }
  return `<svg viewBox="0 0 ${n} ${h}" preserveAspectRatio="none" shape-rendering="crispEdges" aria-hidden="true">${rects}</svg>`;
}

function eegSvg(values, bounds, start, end) {
  const n = end - start;
  const line = (series, cls) => {
    const [low, high] = bounds[series];
    let d = "";
    for (let i = start; i < end; i++) {
      const y = 58 - 52 * (values[series][i] - low) / ((high - low) || 1);
      d += (i === start ? "M" : "L") + (i - start + 0.5) + " " + y.toFixed(1);
    }
    return `<path class="${cls}" d="${d}" fill="none" vector-effect="non-scaling-stroke"/>`;
  };
  return `<svg viewBox="0 0 ${n} 64" preserveAspectRatio="none" aria-hidden="true">${line("delta", "eeg-delta")}${line("beta", "eeg-beta")}</svg>`;
}

function matrixTable({ rows, columns, cell, rowHead, colHead, mark }) {
  const esc = kit.esc;
  const head = columns.map(c => `<th scope="col">${stageDot(c)}${SLEEP_STAGES[c]}</th>`).join("");
  const body = rows.map(r => `<tr><th scope="row">${stageDot(r)}${SLEEP_STAGES[r]}</th>${columns.map(c => {
    const { strength, text, title } = cell(r, c);
    const k = Math.max(0, Math.min(strength, 1));
    const role = r === c ? "diag" : mark === "hit" ? "off" : "flow";
    const cls = [role, k < 0.045 ? "nil" : "", role !== "flow" && k >= 0.62 ? "hi" : ""].filter(Boolean).join(" ");
    return `<td class="${cls}" style="--k:${(k * 100).toFixed(0)}%" title="${esc(title)}">${text}</td>`;
  }).join("")}</tr>`).join("");
  return `<table class="sx-matrix">
    <caption class="sx-sr">${esc(rowHead)} against ${esc(colHead)}</caption>
    <colgroup><col class="stub">${"<col>".repeat(columns.length)}</colgroup>
    <thead>
      <tr><th class="stub" rowspan="2" scope="row">${esc(rowHead)}</th><th colspan="${columns.length}" scope="colgroup">${esc(colHead)}</th></tr>
      <tr>${head}</tr>
    </thead><tbody>${body}</tbody></table>`;
}

function hitNote(confusion) {
  let worst = 0, worstHit = 2, topC = 0, top = -1, seen = false;
  for (let r = 0; r < confusion.length; r++) {
    const total = confusion[r].reduce((a, b) => a + b, 0);
    if (!total) continue;
    const hit = confusion[r][r] / total;
    if (hit < worstHit) {
      worstHit = hit; worst = r; top = -1; seen = true;
      for (let c = 0; c < confusion.length; c++) if (c !== r && confusion[r][c] > top) { top = confusion[r][c]; topC = c; }
    }
  }
  if (!seen) return "No scored segments.";
  const wrong = Math.round((1 - worstHit) * 100);
  if (wrong < 8) return "Almost every stage is read correctly.";
  return `${SLEEP_STAGES[worst]} is the miss (${wrong}% wrong), mostly read as ${SLEEP_STAGES[topC]}.`;
}

function f1Note(hmm, ind) {
  let low = 0, gainAt = 0;
  for (let s = 1; s < hmm.length; s++) {
    if (hmm[s] < hmm[low]) low = s;
    if (hmm[s] - ind[s] > hmm[gainAt] - ind[gainAt]) gainAt = s;
  }
  const gain = hmm[gainAt] - ind[gainAt];
  const weak = gainAt === low && gain >= 0.02
    ? `${SLEEP_STAGES[low]} is weakest: HMM ${hmm[low].toFixed(2)}, independent ${ind[low].toFixed(2)}`
    : `${SLEEP_STAGES[low]} is weakest (${hmm[low].toFixed(2)})`;
  if (gainAt === low || gain < 0.02) return weak + ".";
  return `${weak}. Largest HMM gain is ${SLEEP_STAGES[gainAt]} (+${gain.toFixed(2)}).`;
}

function stayNote(transition) {
  const stay = transition.map((row, i) => row[i]);
  const lo = Math.round(Math.min(...stay) * 100), hi = Math.round(Math.max(...stay) * 100);
  return `Stages usually continue (${lo}–${hi}% are still the same 30 s later).`;
}

function mixNote(counts) {
  const total = counts.reduce((a, b) => a + b, 0) || 1;
  let top = 0;
  counts.forEach((n, i) => { if (n > counts[top]) top = i; });
  const share = Math.round(counts[top] / total * 100);
  return top === 0
    ? `Wake is ${share}% of training, so plain accuracy looks easy.`
    : `${SLEEP_STAGES[top]} is ${share}% of training.`;
}

function buildSleepExplorer(q, i, unit, tools) {
  kit = tools;
  const { DATASETS, ICON_PLAY, esc, highlight, datasetPanelMarkup, wireDataset } = kit;
  const dataset = DATASETS[unit];
  const art = document.createElement("article");
  art.className = "q sx"; art.id = "q" + (i + 1); art.hidden = true;
  art.setAttribute("role", "tabpanel");
  art.innerHTML = `
    <div class="q-heading">
      <p class="q-tag">${esc(q.tag)}</p>
      <button class="btn dataset-button" type="button" aria-expanded="false" aria-controls="dataset-${i + 1}">Dataset</button>
    </div>
    <h1 class="q-title">${esc(q.title)}</h1>
    <p class="q-desc">Pick a person and press <b>Run</b>. Python trains a Hidden Markov Model on the <b>other seven</b> people's labelled nights, then reads the chosen person's recording without seeing their expert labels, and predicts a sleep stage for every 30 seconds. Compare it with the expert, and with a model that ignores the order of the night.</p>
    ${datasetPanelMarkup(i, dataset)}
    <section class="sx-people" aria-labelledby="sx-people-${i}">
      <div class="sx-people-head">
        <div><h2 id="sx-people-${i}">Who should be tested?</h2>
          <p class="sx-legend"><i class="sx-pip test"></i> tested person <i class="sx-pip"></i> trained on</p></div>
        <button class="btn sx-all" type="button">Run all 8</button>
      </div>
      <div class="sx-grid">${SLEEP_SUBJECTS.map(s => `
        <div class="sx-card" data-subject="${s}">
          <div class="sx-card-top"><b>${s}</b><span class="sx-pips" aria-hidden="true">${SLEEP_SUBJECTS.map(t => `<i class="sx-pip${t === s ? " test" : ""}"></i>`).join("")}</span></div>
          <p class="sx-state">Not run yet</p>
          <button class="btn sx-run" type="button" aria-label="Run ${s}: train on the other seven, test on ${s}">${ICON_PLAY} Run</button>
        </div>`).join("")}
      </div>
    </section>
    <div class="sx-result" aria-live="polite"></div>
    <details class="sx-code"><summary>Python that runs on every click</summary><pre></pre></details>`;
  wireDataset(art, dataset);

  const $ = s => art.querySelector(s);
  const grid = $(".sx-grid"), resultBox = $(".sx-result"), runAll = $(".sx-all");
  const results = {};                                   // subject → decoded fold, kept so a person is only run once
  const cards = {};
  SLEEP_SUBJECTS.forEach(s => { cards[s] = { el: grid.querySelector(`[data-subject="${s}"]`) }; });
  let selected = null, running = false, failure = null;

  const codeBox = $(".sx-code");
  codeBox.addEventListener("toggle", async () => {
    if (!codeBox.open) return;
    const pre = codeBox.querySelector("pre");
    try {
      sleepSource = sleepSource || sleepModelCode(await (await fetch("code/week4.py")).text());
      pre.innerHTML = highlight(sleepSource);
    } catch (e) { pre.textContent = "Could not load the code: " + e.message; }
  });

  // ---- cards ----
  function paintCards(active) {
    SLEEP_SUBJECTS.forEach(s => {
      const card = cards[s].el, result = results[s];
      card.classList.toggle("selected", s === selected);
      card.classList.toggle("running", s === active);
      card.classList.toggle("has-result", Boolean(result));
      card.querySelector(".sx-state").textContent = s === active ? "Training on the other seven…"
        : result ? `HMM ${pct(result.models.decoded.accuracy)} · F1 ${result.models.decoded.macro_f1.toFixed(3)}` : "Not run yet";
      const button = card.querySelector(".sx-run");
      button.disabled = running;
      button.innerHTML = ICON_PLAY + (s === active ? " Running…" : result ? " Run again" : " Run");
    });
    runAll.disabled = running;
  }

  // Everything locks synchronously on the first click, so rapid clicks cannot start a second run or
  // show a stale result under another person's name.
  async function start(subjects) {
    if (running) return;
    running = true; failure = null;
    try {
      for (const subject of subjects) {
        paintCards(subject);
        try { results[subject] = await runSleepFold(subject); selected = subject; }
        catch (e) { failure = { subject, message: String(e && e.message || e) }; break; }
        render();
      }
    } finally { running = false; paintCards(null); render(); }
  }
  grid.addEventListener("click", e => {
    const card = e.target.closest(".sx-card");
    if (!card) return;
    const subject = card.dataset.subject;
    if (e.target.closest(".sx-run")) start([subject]);
    else if (results[subject] && !running) { selected = subject; paintCards(null); render(); }
  });
  runAll.addEventListener("click", () => start(SLEEP_SUBJECTS));
  paintCards(null);

  // ---- result ----
  let view = null;                                      // timeline state of the shown person
  let timeline = null;
  function render() {
    if (timeline) timeline.disconnect();
    timeline = null;
    resultBox.replaceChildren();
    if (failure) {
      resultBox.innerHTML = `<div class="sx-error"><b>Could not run ${esc(failure.subject)}.</b> ${esc(failure.message)}</div>`;
    }
    if (!selected || !results[selected]) return;
    const data = results[selected];
    const section = document.createElement("div");
    section.className = "sx-out";
    resultBox.appendChild(section);
    section.innerHTML = summaryHtml(data) + `<div class="sx-timeline"></div>` + detailsHtml(data) + peopleHtml();
    view = { size: 3 * SEGMENTS_PER_HOUR, start: 0, cursor: null, mistakes: true };
    view.size = Math.min(view.size, data.segments);
    view.start = Math.round((data.segments - view.size) / 2);   // a fixed rule that never looks at labels: the middle 3 hours
    timeline = buildTimeline(section.querySelector(".sx-timeline"), data, view);
    section.querySelectorAll("[data-pick]").forEach(row => {
      const pick = () => { if (!running) { selected = row.dataset.pick; paintCards(null); render(); } };
      row.onclick = pick;
      row.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
    });
  }

  function summaryHtml(data) {
    const decoded = data.models.decoded, independent = data.models.independent;
    const dAcc = (decoded.accuracy - independent.accuracy) * 100, dF1 = decoded.macro_f1 - independent.macro_f1;
    const signed = (v, digits) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
    const flips = { expert: switches(data.expert), independent: switches(data.independent), decoded: switches(data.decoded) };
    return `
      <div class="sx-head">
        <h2>${esc(data.subject)} <span>tested · ${(data.segments * 30 / 3600).toFixed(1)} h · ${data.scored_segments.toLocaleString("en-US")} scored segments</span></h2>
        <p>Model trained on ${data.training_subjects.map(esc).join(", ")} (${data.training_segments.toLocaleString("en-US")} labelled segments).
        ${esc(data.subject)}'s expert labels were used only to score.</p>
      </div>
      <div class="sx-score">
        ${SLEEP_MODELS.map(m => `
          <div class="sx-model${m.key === "decoded" ? " best" : ""}">
            <div class="sx-model-name"><b>${m.name}</b><span>${m.note}</span></div>
            <div class="sx-metric"><label>Accuracy</label><span class="sx-bar"><i style="width:${data.models[m.key].accuracy * 100}%"></i></span><b>${pct(data.models[m.key].accuracy)}</b></div>
            <div class="sx-metric"><label>Macro F1</label><span class="sx-bar"><i style="width:${data.models[m.key].macro_f1 * 100}%"></i></span><b>${data.models[m.key].macro_f1.toFixed(3)}</b></div>
            ${m.key === "majority" ? "" : `<div class="sx-flips">${(m.key === "decoded" ? flips.decoded : flips.independent)} stage switches <small>(expert ${flips.expert})</small></div>`}
          </div>`).join("")}
      </div>
      <p class="sx-delta">HMM vs independent: <b>${signed(dAcc, 1)} points</b> accuracy, <b>${signed(dF1, 3)}</b> macro F1. The gain comes from knowing that sleep stages last for minutes rather than flipping every 30 seconds.</p>`;
  }

  function detailsHtml(data) {
    const hmm = data.models.decoded, ind = data.models.independent;
    const stages = [0, 1, 2, 3, 4];
    const confusion = matrixTable({
      rows: stages, columns: stages, rowHead: "Expert", colHead: "Predicted", mark: "hit",
      cell: (r, c) => {
        const total = hmm.confusion[r].reduce((a, b) => a + b, 0), count = hmm.confusion[r][c];
        const share = total ? count / total : 0;
        return { strength: share, text: total ? Math.round(share * 100) + "%" : "–", title: `${count} of ${total} expert ${SLEEP_STAGES[r]} segments predicted as ${SLEEP_STAGES[c]}` };
      },
    });
    const scoreCell = (v, kind) => `<span class="score"><b>${v.toFixed(2)}</b><i class="${kind}" style="--k:${(v * 100).toFixed(0)}%"></i></span>`;
    const f1 = `<table class="sx-cmp"><colgroup><col class="stage"><col><col></colgroup>
      <thead><tr><th scope="col">Stage</th><th scope="col">HMM</th><th scope="col">Independent</th></tr></thead>
      <tbody>${stages.map(s => `<tr><th scope="row">${stageDot(s)}${SLEEP_STAGES[s]}</th><td>${scoreCell(hmm.stage_f1[s], "hmm")}</td><td>${scoreCell(ind.stage_f1[s], "ind")}</td></tr>`).join("")}</tbody></table>`;
    const transition = matrixTable({
      rows: stages, columns: stages, rowHead: "Now", colHead: "30 s later", mark: "flow",
      cell: (r, c) => {
        const p = data.transition[r][c];
        return { strength: p, text: p.toFixed(2), title: `P(${SLEEP_STAGES[c]} next | ${SLEEP_STAGES[r]} now) = ${p.toFixed(4)}` };
      },
    });
    const groups = ["EEG Fpz-Cz", "EEG Pz-Oz", "Eye (EOG)"], bands = ["δ", "θ", "α", "σ", "β"];
    const signature = `
      <table class="sx-matrix sig"><caption>What each stage looks like to the model: average of each input in training (standardised; warm = above average, cool = below)</caption>
        <thead><tr><th></th>${groups.map(g => `<th colspan="5" scope="colgroup">${g}</th>`).join("")}</tr>
        <tr><th></th>${data.features.map((_, k) => `<th scope="col">${bands[k % 5]}</th>`).join("")}</tr></thead>
        <tbody>${stages.map(s => `<tr><th scope="row">${stageDot(s)}${SLEEP_STAGES[s]}</th>${data.signature[s].map((v, k) => {
          const strength = Math.min(Math.abs(v) / 1.6, 1);
          return `<td class="${v >= 0 ? "pos" : "neg"}" style="--k:${(strength * 100).toFixed(0)}%" title="${esc(data.features[k])}: ${v.toFixed(2)}">${v.toFixed(1)}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table>`;
    const counts = data.training_stage_counts, total = counts.reduce((a, b) => a + b, 0) || 1;
    const mix = `<table class="sx-mix"><colgroup><col class="stage"><col></colgroup>
      <thead><tr><th scope="col">Stage</th><th scope="col">Share</th></tr></thead>
      <tbody>${SLEEP_STAGES.map((name, s) => `<tr><th scope="row">${stageDot(s)}${name}</th>
        <td><span class="share"><i class="s${s}" style="--k:${(counts[s] / total * 100).toFixed(1)}%"></i><b>${(counts[s] / total * 100).toFixed(1)}%</b></span></td></tr>`).join("")}</tbody></table>`;
    return `
      <div class="sx-grid2">
        <section class="sx-card2"><h3>HMM vs expert</h3><p class="sx-kicker">Each row is 100%</p>${confusion}
          <p class="sx-note">${esc(hitNote(hmm.confusion))}</p></section>
        <section class="sx-card2"><h3>Per-stage F1</h3>${f1}
          <p class="sx-note">${esc(f1Note(hmm.stage_f1, ind.stage_f1))}</p></section>
      </div>
      <details class="sx-learned" open><summary>What the model learned from the other seven people</summary>
        <div class="sx-grid2">
          <section class="sx-card2"><h3>Stage transitions</h3><p class="sx-kicker">Each row sums to 1</p>${transition}
            <p class="sx-note">${esc(stayNote(data.transition))}</p></section>
          <section class="sx-card2"><h3>Training mix</h3>${mix}
            <p class="sx-note">${esc(mixNote(counts))}</p></section>
        </div>
        <section class="sx-card2 wide"><h3>Stage signatures</h3>${signature}</section>
      </details>`;
  }

  function peopleHtml() {
    const done = SLEEP_SUBJECTS.filter(s => results[s]);
    if (done.length < 1) return "";
    const pooled = {};
    SLEEP_MODELS.forEach(m => {
      pooled[m.key] = scoreOf(done.map(s => results[s].models[m.key].confusion).reduce(addMatrices));
    });
    const label = m => m.key === "majority" ? "Baseline" : m.key === "independent" ? "Independent" : "HMM";
    const cell = (text, i) => `<td${i === 3 ? ' class="split"' : ""}>${text}</td>`;
    const metrics = (accuracy, f1) => accuracy.concat(f1).map(cell).join("");
    const heads = split => SLEEP_MODELS.map((m, i) => `<th scope="col"${split && i === 0 ? ' class="split"' : ""}>${label(m)}</th>`).join("");
    const pending = `<tr class="todo"><th scope="row">{id}</th><td colspan="3">not run</td><td class="split" colspan="3">not run</td></tr>`;
    const rows = SLEEP_SUBJECTS.map(s => results[s] ? `
      <tr data-pick="${s}" class="${s === selected ? "on" : ""}" tabindex="0" role="button" aria-label="Show ${s}">
        <th scope="row">${s}</th>
        ${metrics(
          SLEEP_MODELS.map(m => pct(results[s].models[m.key].accuracy)),
          SLEEP_MODELS.map(m => results[s].models[m.key].macro_f1.toFixed(3))
        )}
      </tr>` : pending.replace("{id}", s)).join("");
    return `
      <section class="sx-people-table"><h3>Across people <span>pooled over ${done.length} of 8${done.length === 8 ? " — the full leave-one-person-out result" : " — run the rest for the full result"}</span></h3>
        <div class="sx-scroll"><table>
          <colgroup><col class="who">${"<col>".repeat(6)}</colgroup>
          <thead>
            <tr><th class="who" rowspan="2" scope="col">Person</th><th colspan="3" scope="colgroup">Accuracy</th><th class="split" colspan="3" scope="colgroup">Macro F1</th></tr>
            <tr>${heads(false)}${heads(true)}</tr>
          </thead>
          <tbody>${rows}</tbody>
          <tfoot><tr><th scope="row">Pooled</th>
            ${metrics(
              SLEEP_MODELS.map(m => pct(pooled[m.key].accuracy)),
              SLEEP_MODELS.map(m => pooled[m.key].macroF1.toFixed(3))
            )}</tr></tfoot>
        </table></div>
        <p class="sx-note">Click a row to look at that person. Each row came from its own model that never saw that person.</p>
      </section>`;
  }

  return art;
}

// ---- the interactive night: hypnogram strips, EEG, stage confidence, one shared cursor ----
function buildTimeline(box, data, view) {
  const n = data.segments;
  const bounds = {}, values = { delta: data.eeg_delta, beta: data.eeg_beta };
  for (const key of ["delta", "beta"]) bounds[key] = [Math.min(...values[key]), Math.max(...values[key])];
  box.innerHTML = `
    <div class="sx-tools">
      <h3>The night, 30 seconds at a time</h3>
      <div class="sx-seg" role="group" aria-label="How much of the night to show">
        ${SLEEP_WINDOWS.map(([label, hours]) => `<button type="button" data-h="${hours}">${label}</button>`).join("")}
      </div>
      <label class="sx-check"><input type="checkbox" class="sx-miss" checked> Mark mistakes</label>
    </div>
    <div class="sx-keys">${SLEEP_STAGES.map((name, s) => `<span>${stageDot(s)}${name}</span>`).join("")}<span><i class="sx-dot miss"></i>mistake</span></div>
    <div class="sx-chart" tabindex="0" role="group" aria-label="Sleep stages over time. Move the pointer, or use the left and right arrow keys, to read each 30-second segment.">
      <div class="sx-row"><span class="lab">Expert</span><div class="plot expert"></div></div>
      <div class="sx-row"><span class="lab">Independent</span><div class="plot independent"></div></div>
      <div class="sx-row"><span class="lab">HMM</span><div class="plot decoded"></div></div>
      <div class="sx-row"><span class="lab">EEG</span><div class="plot eeg"></div></div>
      <div class="sx-row"><span class="lab">Confidence</span>
        <div class="plot heat"><span class="bands" aria-hidden="true">${SLEEP_STAGES.map(s => `<i>${s}</i>`).join("")}</span><canvas></canvas></div></div>
      <div class="sx-row axis"><span class="lab"></span><div class="plot ticks"></div></div>
      <div class="sx-cursor" hidden></div>
    </div>
    <input class="sx-pan" type="range" min="0" step="1" aria-label="Move the window along the night">
    <div class="sx-read" aria-live="off"></div>
    <p class="sx-window"></p>
    <p class="sx-eeg-key"><i class="eeg-delta"></i>Slow waves <i class="eeg-beta"></i>Fast waves</p>`;
  const $ = s => box.querySelector(s);
  const chart = $(".sx-chart"), cursorEl = $(".sx-cursor"), read = $(".sx-read"), pan = $(".sx-pan");
  const canvas = $("canvas"), mistakeBox = $(".sx-miss");
  const plots = { expert: $(".plot.expert"), independent: $(".plot.independent"), decoded: $(".plot.decoded"), eeg: $(".plot.eeg"), ticks: $(".plot.ticks") };

  const end = () => view.start + view.size;
  const clampStart = () => { view.start = Math.max(0, Math.min(n - view.size, view.start)); };

  function drawHeat() {
    const g = canvas.getContext && canvas.getContext("2d");
    if (!g) return;
    const style = getComputedStyle(document.documentElement), width = view.size;
    canvas.width = width; canvas.height = 5;
    for (let s = 0; s < 5; s++) {
      g.fillStyle = style.getPropertyValue("--st" + s).trim() || "#888";
      for (let k = 0; k < width; k++) {
        const p = data.posterior[view.start + k][s];
        if (p < 0.02) continue;
        g.globalAlpha = p; g.fillRect(k, s, 1, 1);
      }
    }
    g.globalAlpha = 1;
  }

  function drawTicks() {
    const from = view.start * 30 / 3600, span = view.size * 30 / 3600;
    const room = Math.max(3, Math.floor((plots.ticks.getBoundingClientRect().width || 600) / 80));
    const steps = [0.1, 0.25, 0.5, 1, 2], step = steps.find(v => span / v <= room) || 2;
    let html = "";
    for (let t = Math.ceil(from / step) * step; t <= from + span + 1e-9; t += step) {
      const at = (t - from) / span * 100;
      html += `<span class="${at > 93 ? "end" : at < 7 ? "start" : ""}" style="left:${at.toFixed(2)}%">${clock(t)}</span>`;
    }
    plots.ticks.innerHTML = html;
  }

  function draw() {
    clampStart();
    const a = view.start, b = end();
    plots.expert.innerHTML = stripSvg(data.expert, null, a, b, false);
    plots.independent.innerHTML = stripSvg(data.independent, data.expert, a, b, view.mistakes);
    plots.decoded.innerHTML = stripSvg(data.decoded, data.expert, a, b, view.mistakes);
    plots.eeg.innerHTML = eegSvg(values, bounds, a, b);
    drawHeat(); drawTicks();
    pan.max = Math.max(0, n - view.size); pan.value = view.start; pan.disabled = view.size >= n;
    box.querySelectorAll(".sx-seg button").forEach(button => {
      const hours = Number(button.dataset.h), size = hours ? hours * SEGMENTS_PER_HOUR : n;
      button.setAttribute("aria-pressed", String(Math.min(size, n) === view.size));
    });
    let wrongInd = 0, wrongHmm = 0, scored = 0;
    for (let k = a; k < b; k++) {
      if (data.expert[k] < 0) continue;
      scored++;
      if (data.independent[k] !== data.expert[k]) wrongInd++;
      if (data.decoded[k] !== data.expert[k]) wrongHmm++;
    }
    $(".sx-window").innerHTML = `<b>${clock(a * 30 / 3600)}–${clock(b * 30 / 3600)}</b> of ${clock(n * 30 / 3600)} h · independent wrong on <b>${wrongInd}</b> of ${scored}, HMM on <b>${wrongHmm}</b>`;
    showCursor();
  }

  function showCursor() {
    const k = view.cursor;
    if (k === null || k < view.start || k >= end()) {
      cursorEl.hidden = true;
      read.innerHTML = `<span class="hint">Hover the chart, or press ← →.</span>`;
      return;
    }
    const column = plots.expert.getBoundingClientRect(), host = chart.getBoundingClientRect();
    const frac = (k - view.start + 0.5) / view.size;
    cursorEl.hidden = false;
    cursorEl.style.left = (column.left - host.left + column.width * frac) + "px";
    const posterior = data.posterior[k];
    const mark = (name, s) => `<span class="${s >= 0 && s !== data.expert[k] && data.expert[k] >= 0 ? "bad" : ""}">${name} ${stageDot(s)}<b>${stageName(s)}</b></span>`;
    read.innerHTML = `<b class="t">${clock(k * 30 / 3600)}</b>
      ${mark("Expert", data.expert[k])}${mark("Independent", data.independent[k])}${mark("HMM", data.decoded[k])}
      <span class="post">${SLEEP_STAGES.map((name, s) => `<span title="${name}: ${(posterior[s] * 100).toFixed(0)}%"><i class="s${s}" style="height:${Math.max(3, posterior[s] * 32)}px"></i></span>`).join("")}</span>`;
  }

  const indexAt = clientX => {
    const rect = plots.expert.getBoundingClientRect();
    const frac = (clientX - rect.left) / rect.width;
    return frac < 0 || frac > 1 ? null : view.start + Math.min(view.size - 1, Math.floor(frac * view.size));
  };
  chart.addEventListener("pointermove", e => { view.cursor = indexAt(e.clientX); showCursor(); });
  chart.addEventListener("pointerleave", () => { view.cursor = null; showCursor(); });
  chart.addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 20 : 1);
    view.cursor = Math.max(0, Math.min(n - 1, (view.cursor === null ? view.start + (view.size >> 1) : view.cursor) + step));
    if (view.cursor < view.start) view.start = view.cursor;
    if (view.cursor >= end()) view.start = view.cursor - view.size + 1;
    draw();
  });
  pan.addEventListener("input", () => { view.start = Number(pan.value); view.cursor = null; draw(); });
  mistakeBox.addEventListener("change", () => { view.mistakes = mistakeBox.checked; draw(); });
  box.querySelectorAll(".sx-seg button").forEach(button => button.onclick = () => {
    const hours = Number(button.dataset.h), centre = view.start + view.size / 2;
    view.size = Math.min(n, hours ? hours * SEGMENTS_PER_HOUR : n);
    view.start = Math.round(centre - view.size / 2); view.cursor = null;
    draw();
  });
  // Redraw the confidence picture when the light/dark theme changes (it is drawn from the theme's colours).
  const watch = new MutationObserver(drawHeat);
  watch.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  draw();
  return watch;
}
