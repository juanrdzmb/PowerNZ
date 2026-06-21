// Loaded synchronously in <head> so the theme is set before the first paint
// (no flash). The rest of the theme logic lives in app.js.
(function () {
  // Marks that JS is active so reveal-on-scroll only hides content when it can
  // also un-hide it. Without JS, [data-reveal] stays fully visible.
  document.documentElement.classList.add("js");
  try {
    var saved = localStorage.getItem("pnz-theme");
    if (saved !== "light" && saved !== "dark") {
      saved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }
    document.documentElement.setAttribute("data-theme", saved);
  } catch (error) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
