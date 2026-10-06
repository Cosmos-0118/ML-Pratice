const assert = require("node:assert/strict");
const { readFileSync, readdirSync, existsSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const root = join(__dirname, "..");
const formatter = join(root, "website/output.js");
const context = vm.createContext({});
if (existsSync(formatter)) vm.runInContext(readFileSync(formatter, "utf8"), context);
const format = text => {
  assert.equal(typeof context.formatOutput, "function", "the raw-output formatter must exist");
  return context.formatOutput(text);
};
const plain = html => html.replace(/<[^>]*>/g, "")
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

test("highlights metric labels and signed, decimal, and scientific values", () => {
  const html = format("Test accuracy: 0.833\nWeight: -0.05\nValues: [.5 1. -2e-3 +4 nan inf]\n");
  assert.match(html, /class="out-label">Test accuracy:<\/span>/);
  for (const value of ["0.833", "-0.05", ".5", "1.", "-2e-3", "+4", "nan", "inf"]) {
    assert.ok(html.includes(`class="hl-num">${value}</span>`), value);
  }
  assert.equal(plain(html), "Test accuracy: 0.833\nWeight: -0.05\nValues: [.5 1. -2e-3 +4 nan inf]\n");
});

test("keeps dates and dtype names whole and preserves table alignment", () => {
  const raw = "First 10 rows:\n         date  temp_max  weather\n0  2012-01-01      -1.0     rain\ndtype: float64\n";
  const html = format(raw);
  assert.match(html, /class="hl-str">2012-01-01<\/span>/);
  assert.match(html, /class="hl-kw">float64<\/span>/);
  assert.match(html, /class="out-header">         date  temp_max  weather<\/span>/);
  assert.equal(plain(html), raw);
});

test("treats printed HTML as text, including entities and quoted attributes", () => {
  const raw = '<script>alert("x")</script>\n<img src=x onerror="alert(1)"> &lt; & < >\n';
  const html = format(raw);
  assert.ok(!html.includes("<script>"));
  assert.ok(!html.includes("<img"));
  assert.equal(plain(html), raw);
});

test("preserves arbitrary prose, blank lines, indentation, and empty output", () => {
  for (const raw of ["", "\n\n", "  hello\tworld\r\n\n", "|   |--- class: rain\n|--- wind <= 1.95\n", "PC1 temp_max http://example.com\n"]) {
    assert.equal(plain(format(raw)), raw);
  }
});

test("formats every saved practice-program output without changing any text", () => {
  let questions = 0;
  for (const file of readdirSync(root).filter(f => /^Week_.*\.ipynb$/.test(f))) {
    const notebook = JSON.parse(readFileSync(join(root, file), "utf8"));
    for (const cell of notebook.cells) {
      const raw = (cell.outputs || []).filter(o => o.output_type === "stream")
        .map(o => Array.isArray(o.text) ? o.text.join("") : o.text).join("");
      if (!raw) continue;
      const html = format(raw);
      assert.equal(plain(html), raw, file);
      assert.match(html, /class="hl-num"/, file);
      questions++;
    }
  }
  assert.equal(questions, 9);
});
