(() => {
  // ---------- Theme toggle ----------
  const root = document.documentElement;
  const themeToggle = document.querySelector('[data-theme-toggle]');
  themeToggle?.addEventListener('click', () => {
    const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('pnz-theme', next); } catch (_) { /* private mode */ }
  });

  // ---------- Privacy banner ----------
  const banner = document.querySelector('[data-privacy-banner]');
  if (banner) {
    let acknowledged = false;
    try { acknowledged = localStorage.getItem('pnz-privacy-ack') === '1'; } catch (_) { acknowledged = false; }
    if (!acknowledged) {
      window.setTimeout(() => banner.classList.add('is-visible'), 450);
    }
    banner.querySelector('[data-privacy-accept]')?.addEventListener('click', () => {
      banner.classList.remove('is-visible');
      try { localStorage.setItem('pnz-privacy-ack', '1'); } catch (_) { /* private mode */ }
    });
  }

  // ---------- Upload affordances ----------
  const fileInput = document.querySelector('#video-input');
  const fileMeta = document.querySelector('[data-file-meta]');
  const dropZone = document.querySelector('[data-drop-zone]');

  function showFile(file) {
    if (!file || !fileMeta) return;
    const megabytes = (file.size / 1024 / 1024).toFixed(file.size > 100 * 1024 * 1024 ? 0 : 1);
    fileMeta.textContent = `${file.name} · ${megabytes} MB`;
    dropZone?.classList.add('has-file');
  }

  fileInput?.addEventListener('change', () => showFile(fileInput.files?.[0]));
  dropZone?.addEventListener('dragover', (event) => { event.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone?.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = event.dataTransfer?.files?.[0];
    if (!file || !fileInput) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showFile(file);
  });

  const analysisForm = document.querySelector('.analysis-form');
  analysisForm?.addEventListener('submit', () => {
    const submit = analysisForm.querySelector('button[type="submit"]');
    if (!submit || !fileInput?.files?.length) return;
    submit.disabled = true;
    const label = submit.querySelector('span');
    if (label) label.textContent = 'Subiendo el vídeo…';
  });

  // ---------- Job status polling ----------
  const statusRoot = document.querySelector('[data-job-status]');
  if (!statusRoot) return;

  const orbit = document.querySelector('.progress-orbit');
  const progress = document.querySelector('[data-progress]');
  const progressBar = document.querySelector('[data-progress-bar]');
  const stage = document.querySelector('[data-stage]');
  const stageHeading = document.querySelector('[data-stage-heading]');
  const detail = document.querySelector('[data-status-detail]');
  const statusLabel = document.querySelector('[data-status-label]');

  function setProgress(value) {
    const pct = Math.max(0, Math.min(100, Number(value) || 0));
    if (orbit) orbit.style.setProperty('--p', pct);
    if (progress) progress.textContent = `${pct}%`;
    if (progressBar) progressBar.value = pct;
  }

  // Animate the ring from its server-rendered value on first paint.
  setProgress(progressBar ? progressBar.value : 0);

  if (statusRoot.getAttribute('data-terminal') === 'true') return;
  const statusUrl = statusRoot.getAttribute('data-status-url');
  let finished = false;

  async function poll() {
    if (finished || !statusUrl) return;
    try {
      const response = await fetch(statusUrl, { credentials: 'same-origin', cache: 'no-store' });
      if (!response.ok) return;
      const data = await response.json();
      setProgress(data.progress);
      if (stage) stage.textContent = data.stage;
      if (stageHeading) stageHeading.textContent = data.stage;
      if (data.terminal) {
        finished = true;
        if (statusLabel) statusLabel.textContent = data.status === 'completed' ? 'COMPLETADO' : 'NECESITA OTRO INTENTO';
        if (detail && data.status === 'failed') detail.textContent = data.error_message;
        window.setTimeout(() => window.location.reload(), 700);
      }
    } catch (_) {
      // A brief network hiccup should not break the page; retry on the next tick.
    }
  }
  poll();
  window.setInterval(poll, 2000);
})();
