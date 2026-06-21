// Loaded synchronously in <head> so the theme is set before the first paint
// (no flash). The rest of the theme logic lives in app.js.
(function () {
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
