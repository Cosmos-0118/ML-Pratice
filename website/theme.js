// Light / dark theme: follows the system until the visitor picks one, then remembers the choice.
// Loaded in <head> so the theme is set before the first paint (no flash).
(function () {
  var root = document.documentElement, key = "theme", mq = matchMedia("(prefers-color-scheme: dark)");
  function saved() { try { return localStorage.getItem(key); } catch (e) { return null; } }
  function apply(t) { root.setAttribute("data-theme", t); }
  apply(saved() || (mq.matches ? "dark" : "light"));
  mq.addEventListener("change", function (e) { if (!saved()) apply(e.matches ? "dark" : "light"); });
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".theme-toggle")) return;
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    apply(next);
    try { localStorage.setItem(key, next); } catch (err) {}
  });
})();
