(() => {
  const fileInput = document.querySelector('#video-input');
  const fileMeta = document.querySelector('[data-file-meta]');
  const dropZone = document.querySelector('[data-drop-zone]');

  function showFile(file) {
    if (!file || !fileMeta) return;
    const megabytes = (file.size / 1024 / 1024).toFixed(file.size > 100 * 1024 * 1024 ? 0 : 1);
    fileMeta.textContent = `${file.name} · ${megabytes} MB · listo para analizar`;
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
  if (label) label.textContent = 'Subiendo tu vídeo…';
});

  const statusRoot = document.querySelector('[data-job-status]');
  if (!statusRoot) return;
  if (statusRoot.getAttribute('data-terminal') === 'true') return;
  const statusUrl = statusRoot.getAttribute('data-status-url');
  const progress = document.querySelector('[data-progress]');
  const progressBar = document.querySelector('[data-progress-bar]');
  const stage = document.querySelector('[data-stage]');
  const stageHeading = document.querySelector('[data-stage-heading]');
  const detail = document.querySelector('[data-status-detail]');
  const statusLabel = document.querySelector('[data-status-label]');
  const statusCard = document.querySelector('[data-status-card]');
  let finished = false;

  async function poll() {
    if (finished || !statusUrl) return;
    try {
      const response = await fetch(statusUrl, { credentials: 'same-origin', cache: 'no-store' });
      if (!response.ok) return;
      const data = await response.json();
      if (progress) progress.textContent = `${data.progress}%`;
      if (progressBar) progressBar.value = data.progress;
      if (stage) stage.textContent = data.stage;
      if (stageHeading) stageHeading.textContent = data.stage;
      if (data.terminal) {
        finished = true;
        if (statusLabel) statusLabel.textContent = data.status === 'completed' ? 'COMPLETADO' : 'NECESITA OTRO INTENTO';
        if (detail && data.status === 'failed') detail.textContent = data.error_message;
        window.setTimeout(() => window.location.reload(), 700);
      }
    } catch (_) {
      // A short network interruption should not spoil the page; try again on the next tick.
    }
  }
  poll();
  window.setInterval(poll, 2000);
})();
