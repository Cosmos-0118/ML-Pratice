// Presentation only: decorate captured stdout without changing its text or spacing.
// Keep this independent of the Python programs and their execution.
function formatOutput(text) {
  const escape = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const span = (cls, s) => `<span class="${cls}">${escape(s)}</span>`;
  const token = /(\b\d{4}-\d{2}-\d{2}\b)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b(?:float\d*|int\d*|uint\d*|object|str|bool|datetime64|category)\b)|(\b(?:True|False|None)\b)|((?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?!\w)|\b(?:nan|inf)\b)|([\[\](){},]|<=|>=|->|[=<>])|([A-Za-z_]\w*)/g;
  const decorate = s => {
    let html = "", last = 0, match;
    token.lastIndex = 0;
    while ((match = token.exec(s))) {
      html += escape(s.slice(last, match.index));
      const [value, date, string, dtype, keyword, number, punctuation] = match;
      const cls = date || string ? "hl-str" : dtype || keyword ? "hl-kw" : number ? "hl-num" : punctuation ? "out-punct" : "";
      html += cls ? span(cls, value) : escape(value);
      last = token.lastIndex;
    }
    return html + escape(s.slice(last));
  };

  return text.split("\n").map(line => {
    // Table headers contain aligned text columns, not numeric data or tree branches.
    if (/^\s*[#A-Za-z_]/.test(line) && /\S[ \t]{2,}\S/.test(line)
        && !/[\d.:\[\]<>|]/.test(line)) return span("out-header", line);
    const label = line.match(/^(\s*[A-Za-z][^:]*:)(?!\/\/)/);
    return label ? span("out-label", label[0]) + decorate(line.slice(label[0].length)) : decorate(line);
  }).join("\n");
}
