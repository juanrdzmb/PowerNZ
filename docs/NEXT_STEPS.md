# PowerAI — Siguientes pasos (roadmap de traspaso)

> Documento para que cualquier agente/colaborador continúe sin perder el hilo. Resume el
> estado actual, las decisiones de diseño que **NO** hay que romper, el problema raíz
> pendiente y un plan priorizado. Entrenamiento previsto en **Colab/nube**; objetivo
> final **app móvil** tipo Lift App.

## Contexto

PowerAI analiza videos de levantamiento (peso muerto) y dibuja un overlay tipo Lift App:
silueta del atleta, esqueleto, trayectoria de barra, velocidad/ROM y panel. Ya se
arreglaron los bugs visuales (silueta, discos, trayectoria), se rediseñó la
visualización a un estilo profesional, se añadió auto-calibración y se limpió el código
muerto (ver PR https://github.com/juanrdzmb/PowerAI/pull/5, rama
`feature/yolo-custom-training`). El overlay ya incluye multicurva de velocidad, tabla
Con/Vel/Peak/Ecc/Loss, etiquetas Plate/Barbell, drift horizontal y carga estimada.
`python -m pytest` → 108 en verde. La mejora grande que sigue pendiente es **reentrenar el
detector en GPU** para generalizar a todos los ángulos.

Videos de prueba: `C:\Users\Juanda\Documents\entrenamiento\peso_muerto_1..9.mp4`
(+ `Sentadila_1`, `sentadilla_2`). `peso_muerto_8` (móvil, multipersona) es el mejor caso
de validación; `peso_muerto_1` es lateral; `peso_muerto_2` es con el que se entrenó el
detector de barra actual.

## Estado actual y decisiones de diseño (NO romper)

- **Detección de discos/barra** (`detect_objects.py`): primario = YOLO entrenado
  `models/powerai_bar_detector.pt` (clases `plate`, `bar_hub`); fallback = heurística de
  color `MultiColorPlateDetector` (rojo+beige). El modelo entrenado **solo detecta bien en
  `peso_muerto_2`** (se entrenó solo con esos frames) → en el resto tira de la heurística.
- **Atleta = quien sostiene la barra** (`pose.py`, `_bar_owner_index` +
  `_select_pose_person_index`): se prefiere la persona cuya caja contiene el hub de la
  barra. Descarta espectadores. La silueta se ata a esa pose.
- **Silueta** (`segmentation.py`): `YoloSegmentationSegmenter` con modelo propio
  `models/powerai_athlete_seg.pt` si existe; si no, modelo de persona pre-entrenado
  `yolo11s-seg` (fallback `yolo11n-seg`, en `models/`). El post-proceso usa umbral,
  morfología, mayor componente sobre la pose, refinado de borde con `guidedFilter` cuando
  está disponible, mezcla temporal y feather. En frames saltados, `main.py` warpea la
  máscara con flujo óptico para reducir lag.
- **Ancla de barra** (`bar_anchor.py`, `BarAnchorTracker`): `set_pose_hint` sesga hacia el
  disco a la altura de las muñecas; filtros OneEuro afinados. **`BarAnchorConfig.
  refined_plate_rect_scale=0.72`** define el rect que mide la escala — **NO subirlo para
  que el recuadro abrace el disco**: el recuadro dibujado se agranda aparte con
  `OverlayConfig.plate_box_display_scale≈1.36` (`render_overlay.py`). Calibración y dibujo
  están desacoplados a propósito (subir el scale del rect estropea el ROM). Ya hay
  predicción Kalman corta (`source="prediction"`) para no congelar la trayectoria si la
  detección se tapa unos frames.
- **Calibración** (`main.py` + `calibration.py`): cuando hay YOLO entrenado, primero mide
  el diámetro desde la caja cruda `plate` más cercana al ancla seleccionada; si no hay
  suficientes cajas fiables, cae a `estimate_plate_diameter_from_tracker`. Disco real =
  0.45 m (`OLYMPIC_PLATE_DIAMETER_METERS`). `--disable-auto-calibration` fuerza
  `--plate-diameter-px`.
- **Trayectoria** (`main.py` + `render_overlay::_draw_bar_path`): se alimenta del hub del
  disco estable cuando es usable, se reinicia por repetición, EMA de render 0.55. El panel
  muestra drift horizontal en cm.
- **Optimización**: la segmentación se refresca cada 2 frames (`seg_stride` en `main.py`) y
  la máscara se warpea con flujo óptico en frames intermedios.
- **Pipeline de entrenamiento**: `datasets/extract_frames.py` →
  `datasets/autolabel_world.py` (YOLO-World bootstrap) / `annotate_*.py` (manual) →
  `datasets/yolo_dataset.py` (split train/val) → `train_bar_detector.py` (plate+bar_hub,
  `datasets/dataset_bar_2cls.yaml`) y `train_athlete_seg.py` (athlete-seg). Frames crudos,
  labels generadas, caches y pesos base están en `.gitignore`.

## Problema raíz central

La fragilidad restante (recuadros que no abrazan bien el disco en algunos videos,
calibración que sobre-estima ~30% y ROM bajo) viene de que **el detector de barra
entrenado no generaliza** (un solo video) y se cae a la heurística de color, que fusiona
el disco con tonos del fondo. **Reentrenar con datos variados desacopla y arregla a la
vez: detección, fijación del recuadro y exactitud de la calibración.**

---

## Fase actual: antes de montar el modelo

Ahora mismo el trabajo útil es ordenar datos y contratos, no asumir que
`models/powerai_bar_detector.pt` ya existe o generaliza.

1. **Etiquetar solo dos clases v1**: `0=plate` y `1=bar_hub`. El anotador
   `datasets/annotate_bar.py` ya abre por defecto con esas dos clases. Si algún día se
   quieren `bar_sleeve` o `bar_shaft`, será otra versión de dataset/modelo.
2. **Cubrir variedad antes de entrenar**: laterales, diagonales, diferentes discos,
   manos tapando el hub, luz mala, multipersona y frames con discos en el suelo.
3. **Negativos importantes**: si un disco está en el suelo o fondo y no pertenece a la
   barra levantada, no etiquetarlo como `plate`. El tracking ya filtra por altura de
   muñecas, pero el modelo aprende de lo que se etiqueta.
4. **Antes de Colab/GPU**: comprobar que cada imagen etiquetada tiene su `.txt`, que los
   labels solo usan clases `0` y `1`, y revisar previews de cajas antes de entrenar.
5. **Después de entrenar**: descargar `best.pt`, guardarlo como
   `models/powerai_bar_detector.pt`, correr smoke visual y revisar que auto-calibración,
   trayectoria, ROM y velocidad mejoran por la detección, no por tocar escalas.

## Roadmap priorizado

### P0 — Detección robusta (reentrenar `powerai_bar_detector.pt`) — mayor impacto
Desbloquea recuadros que abrazan el disco + calibración exacta + trayectoria estable.
1. Extraer frames de **todos** los videos de `entrenamiento` (no solo peso_muerto_2),
   variando ángulo/luz/color de disco: `python datasets/extract_frames.py` (revisar args;
   apuntar a cada video).
2. Bootstrap opcional de etiquetas con `datasets/autolabel_world.py` (YOLO-World, clases
   `barbell weight plate` y `barbell collar sleeve`) y **corregir a mano** las malas con
   `datasets/annotate_bar.py`. Incluir frames multipersona y con discos en el suelo como
   negativos (que NO se etiquete el disco del suelo como el de la barra). En esta máquina
   no hay CUDA; YOLO-World `x` en CPU quedó demasiado lento, así que completar esta parte
   en Colab/GPU.
3. Entrenar en **Colab** (GPU): subir `datasets/training/{frames,labels}` +
   `dataset_bar_2cls.yaml` y correr el equivalente a
   `python train_bar_detector.py --data datasets/dataset_bar_2cls.yaml --base-model yolov8n.pt --epochs 120 --imgsz 960 --batch 16 --device 0`.
   Traer `best.pt` → `models/powerai_bar_detector.pt`.
4. Validar mAP y revisar `outputs/validation/dataset_label_review/bar_detector`. Objetivo:
   `plate` y `bar_hub` detectados en peso_muerto_1/3/5/7/8.

**Efecto esperado:** el rect del ancla sale de una caja `plate` ajustada → el recuadro
abraza el disco sin el truco de color; y la nueva calibración por caja `plate` mide el
diámetro real (ROM correcto en todos los videos).
Archivos: `datasets/*`, `train_bar_detector.py`. Sin cambios de runtime.

### P1 — Silueta que se ciñe aún mejor (modelo `athlete` propio)
El `yolo11s-seg` genérico (COCO person) ya ciñe bien, pero falla en bisagra de cadera
extrema, oclusión por discos y ropa/equipo del gym.
1. Anotar máscaras de `athlete` (y `background_person` para ignorar fondo) con
   `datasets/annotate_athlete.py`; o bootstrap con SAM/yolo11x-seg y corregir.
2. Entrenar seg en Colab: `python train_athlete_seg.py --data datasets/dataset_athlete.yaml
   --base-model yolo11s-seg.pt --epochs 100 --imgsz 768 --device 0` → traer a
   `models/powerai_athlete_seg.pt`. Usar con `--segmentation-model`.
3. Runtime opcional (sin reentrenar): **implementado**. `guidedFilter` refina el borde si
   está disponible y `main.py` warpea la máscara por flujo óptico en los frames saltados.
Archivos: `datasets/annotate_athlete.py`, `train_athlete_seg.py`, `segmentation.py`.

### P2 — Fijación del recuadro y seguimiento durante el levantamiento
Mayormente sale gratis con P0; extras de robustez:
1. Usar la detección `bar_hub` entrenada como **único punto métrico**. Implementado en
   modo estricto: si hay `plate` pero no `bar_hub`, se dibuja el disco, pero no se mide
   velocidad ni trayectoria.
2. Filtro Kalman/predictor de velocidad constante para oclusión corta:
   **implementado** como `source="prediction"` en `BarAnchorTracker`.
3. Fijar el `track_id` del disco durante toda la repetición (ByteTrack ya activo vía
   `detect_with_tracking`); priorizar el mismo id frame a frame.

### P3 — Trayectoria y métricas (pulido tipo Lift App)
1. Métrica de **desviación de la barra** (drift horizontal en cm): **implementada** desde
   `bar_path` con la calibración y visible en el panel.
2. **Peso estimado** desde el color del disco (gris 5 / verde 10 / amarillo 15 / azul 20 /
   rojo 25 kg): **implementado** en `load_estimation.py` y visible como "Load: X kg" cuando
   hay color fiable.
3. **Pérdida de velocidad/fatiga**: **implementada** como `Loss` en la tabla, usando el %
   respecto a la mejor repetición.
4. Línea de referencia vertical ideal opcional sobre la trayectoria.

### P4 — Optimización y camino a móvil (objetivo final)
1. **Inferencia a resolución reducida**: correr pose/objeto/seg a ~720p y mapear
   coordenadas de vuelta, en vez de 4K. Gran speedup. Ya existe `--max-resolution`;
   considerar hacerlo el default interno de inferencia.
2. **Exportar modelos**: script **implementado** en `export_models.py` para ONNX/TFLite/
   CoreML, con tamaño de artefacto y latencia PyTorch aproximada.
3. Para la app: portar la lógica de tracking/estado (`bar_anchor.py`, `metrics.py`) a la
   plataforma o exponer un núcleo en Python/ONNX-runtime; mantener el overlay como capa de
   UI nativa. Evaluar modelos `n` (nano) por latencia en dispositivo.
4. Reducir modelos: pose `yolo11n-pose`, seg `yolo11n-seg`, detector de barra cuantizado.

---

## Verificación (para cada cambio)
1. Tests: `python -m pytest` (108 deben pasar; actualizar si cambia el contrato).
2. Smoke + inspección visual (extraer frames y abrirlos):
   ```powershell
   python main.py --input "C:\Users\Juanda\Documents\entrenamiento\peso_muerto_8.mp4" `
     --output outputs\check.mp4 --pose-backend yolo --plate-diameter-px 120 `
     --report-json outputs\check.json --no-mobile-conversion
   ```
   Confirmar en el JSON/overlay: ROM ~0.5 m y reps validadas; recuadro abrazando el disco;
   silueta ceñida; trayectoria continua siguiendo la barra; atleta correcto (no el
   espectador). Repetir en peso_muerto_1/3/5/7.
3. Tras reentrenar: comprobar que la calibración (`Auto-calibration observed`) da un
   diámetro coherente y el ROM no se desvía entre videos.

## Caveat clave
El detector de barra actual solo cubre `peso_muerto_2`; **por eso P0 es lo primero** —
desbloquea de un golpe la detección, la fijación del recuadro al disco y la exactitud de
la calibración.
