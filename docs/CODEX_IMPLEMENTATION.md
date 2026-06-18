# Instrucciones para Codex — Continuar PowerAI (detector de discos + pulido tipo Lift App)

## Estado posterior — 2026-06-16

Esta guía ya fue ejecutada en la rama `feature/yolo-custom-training`:
- Vía A implementada y verificada: silueta suavizada con `guidedFilter`/fallback, velocidades multipunto, gráfico multicurva, tabla Con/Vel/Peak/Ecc/Loss, etiquetas `Plate`/`Barbell` y trayectoria por hub.
- Vía B tooling implementado: extracción de frames, `datasets/autolabel_world.py`, `.gitignore` para pesos base y tests. El entrenamiento completo de `models/powerai_bar_detector.pt` queda pendiente de GPU/Colab: en esta máquina `torch.cuda.is_available()` fue `False` y YOLO-World `x` en CPU no terminó en 20 minutos.
- Mejoras extra implementadas donde no dependían de GPU: calibración por caja cruda `plate`, drift horizontal en cm, carga estimada por color de disco, pérdida de velocidad en tabla, auto-uso de `models/powerai_athlete_seg.pt` si existe, `export_models.py` para ONNX/TFLite/CoreML y warp de máscara por flujo óptico.
- Verificación final: `python -m pytest` → 91 passed. Smoke real en `peso_muerto_8` con 320 frames: 320/320 tracked y hub fiable.

> Eres un agente continuando el trabajo en **PowerAI** (CV para analizar peso muerto, overlay tipo
> Lift App). Implementa EXACTAMENTE lo de abajo. **No reinventes la arquitectura ni cambies cosas
> que ya funcionan.** Trabaja en la rama `feature/yolo-custom-training`. Tras cada bloque corre
> `python -m pytest` y haz un smoke con un video real, extrayendo frames para verificar a ojo.
> Repo: `C:\Users\Juanda\Documents\PowerAI`. Videos: `C:\Users\Juanda\Documents\entrenamiento\peso_muerto_1..9.mp4`.

## 0. Estado actual y REGLAS (no romper)
- Pipeline: `main.py` (orquesta) → `pose.py` (YOLO-pose, atleta = quien sostiene la barra via
  `_bar_owner_index`) → `detect_objects.py` (YOLO entrenado `models/powerai_bar_detector.pt` +
  fallback color `MultiColorPlateDetector`) → `bar_anchor.py` (`BarAnchorTracker`, ancla del disco) →
  `metrics.py` (velocidad/ROM/reps) → `anchor_metrics.py` (velocidad de keypoints) → `render_overlay.py`.
- **Decisiones que NO se tocan**:
  - Calibración (`main.py::estimate_plate_diameter_from_tracker`) usa el rect del ancla a escala
    `BarAnchorConfig.refined_plate_rect_scale=0.72`. **No** subas ese scale para agrandar el recuadro:
    el dibujo se agranda aparte con `OverlayConfig.plate_box_display_scale`. Subirlo descuadra el ROM.
  - El recuadro NO debe saltar entre discos: `bar_anchor._select_observation` ya sostiene posición si
    la única detección cercana es lejana. Mantenlo.
  - Filtro de zona-barra `main.py::_filter_detections_near_bar` (descarta discos del suelo/fondo). Mantenlo.
  - Métrica de barra: hub del disco estable; muñeca como respaldo. La silueta se ata a la pose del atleta.
  - Segmentación cada 2 frames (`seg_stride`). Silueta = `yolo11s-seg` (persona) con post-proceso.
- Corre `python -m pytest` antes de empezar (deben pasar ~85). Si cambias un contrato, ACTUALIZA el test.

---

## VÍA A — Pulido de runtime (HACER PRIMERO, no depende de entrenar)

### A1. Silueta suave, sin bordes dentados — `segmentation.py::YoloSegmentationSegmenter._postprocess_mask`
- Tras umbral + morfología + mayor-componente-sobre-pose (ya existe), añade refinado de borde:
  usa `cv2.ximgproc.guidedFilter(guide=frame_gray, src=mask, radius≈ max(8,frame_w*0.01), eps=1e-2)`
  (opencv-contrib YA instalado) y re-umbral suave; si `ximgproc` no está, cae a `MORPH_CLOSE`+`OPEN`
  con kernel mayor. Sube el feather final (Gaussian) ~1.5x. Resultado: contorno liso, ceñido, sin quiebres.
- No cambies el blend del relleno (hay un test que fija el canal azul a `silhouette_alpha=0.50`).

### A2. Velocidad multipunto + gráfico multicurva + tabla — `anchor_metrics.py`, `metrics.py`, `reporting.py`, `render_overlay.py`, `main.py`
1. `anchor_metrics.py`: en `ANCHOR_GROUPS` añade `("shoulder", ("left_shoulder","right_shoulder"))` y
   `("elbow", ("left_elbow","right_elbow"))` (ya hay wrist/hip/knee). Mantén el OneEuro por grupo y la
   selección de lado más visible. Estos son los puntos de anclaje del esqueleto con su velocidad.
2. `main.py`: mantén un `dict[str, list[float]]` `anchor_velocity_history` (clave = nombre del anclaje,
   incl. `"bar"` con la velocidad de la barra). Cada frame con sample válido, hace append de cada
   `AnchorVelocity.velocity_mps` y de `sample.smoothed_velocity_mps` en `"bar"`. Pásalo al renderer.
3. `metrics.py`/`reporting.py`: expón **Con(s)** y **Ecc(s)** por rep. `CompletedRep` ya tiene
   `start_frame`, `lockout_frame`, `end_frame` → `con_s=(lockout-start)/fps`, `ecc_s=(end-lockout)/fps`.
   Añádelos a `RepReport` (`reporting.py`) y rellénalos en `RepReportBuilder.build_rep_report`.
4. `render_overlay.py`: nuevo `_draw_multi_velocity_chart` (sustituye/junto al actual): una curva por
   anclaje con color fijo + **leyenda** (Hip/Knee/Shoulder/Elbow/Bar) como la referencia; bandas de
   fondo alternas por repetición (usa frames de inicio/fin de cada `CompletedRep`). Mantén el panel
   redondeado, sin solapar la tabla (la tabla se apila encima — ya hay `bottom_limit`).
5. Tabla estilo Lift App: columnas **Rep · Con(s) · Vel(m/s) · Peak(m/s) · Ecc(s)** en `_draw_rep_table`.

### A3. Recuadros etiquetados + trayectoria por el disco — `render_overlay.py`
- En `_draw_bar_anchor`: etiqueta el recuadro grande del disco como "Plate" y el cuadro pequeño del hub
  como "Barbell" (pastillas pequeñas, como la referencia). Reutiliza `anchor.rect` y `anchor.hub_rect`.
- Asegura que `bar_path` (trayectoria) arranca en el hub y queda continua y casi vertical (ya hay rechazo
  de saltos horizontales en `main.py` y `_draw_bar_path`). La línea debe verse pasando por el disco.

### Verificación A
`python -m pytest` verde. Smoke en peso_muerto_1 (lateral), _3 (frontal), _8 (multipersona):
```
python main.py --input "C:\Users\Juanda\Documents\entrenamiento\peso_muerto_8.mp4" --output outputs\a.mp4 --pose-backend yolo --plate-diameter-px 120 --report-json outputs\a.json --no-mobile-conversion
```
Extrae frames con cv2 y revisa: silueta lisa; multicurva con leyenda; tabla Con/Vel/Peak/Ecc; recuadros
"Plate"/"Barbell"; trayectoria por el centro del disco; sin solapes.

---

## VÍA B — Entrenar el detector de discos/barra (HAZLO TÚ, Codex)
Meta: `models/powerai_bar_detector.pt` (clases `plate`, `bar_hub`) que generalice a todos los ángulos →
con `track_id` de ByteTrack el recuadro queda FIJO en el disco todo el video.

1. **Extraer frames** de todos los videos:
   `python datasets/extract_frames.py --input C:\Users\Juanda\Documents\entrenamiento\peso_muerto_1.mp4 ... peso_muerto_9.mp4 --max-frames 180` → `datasets/training/frames/`.
2. **Auto-etiquetar** (Codex no puede anotar a mano): escribe `datasets/autolabel_world.py` usando
   `from ultralytics import YOLOWorld; m=YOLOWorld("yolov8x-worldv2.pt"); m.set_classes(["barbell weight plate","barbell collar sleeve"])`
   y guarda labels YOLO (clase 0=plate, 1=bar_hub) por frame en `datasets/training/labels/`. Filtra por
   confianza ≥0.25 y descarta cajas gigantes (>0.8 del frame). Esto da etiquetas mucho mejores que el color.
   (El usuario puede luego corregir las malas con `datasets/annotate_bar.py`.)
3. **Entrenar**: `python train_bar_detector.py --data datasets/dataset_bar_2cls.yaml --base-model yolo11n.pt --epochs 120 --imgsz 960 --batch 8 --device 0` (usa `--device cpu` si no hay GPU; será lento — preferible Colab/GPU). Trae `best.pt` → `models/powerai_bar_detector.pt`.
4. **Validar e integrar**: corre el smoke; confirma que `detect_with_tracking` propaga `track_id` y que
   `bar_anchor._select_observation` lo usa (ya existe). El recuadro debe quedar **fijo** en el disco todo
   el video, en lateral/frontal/diagonal, sin saltar a discos del suelo. Revisa mAP del run en `runs/detect/`.

---

## Mejoras extra (impleméntalas después de A y B, en este orden)
1. **Calibración exacta con el modelo entrenado**: con cajas `plate` ajustadas, mide el diámetro real del
   disco (mediana de la caja `plate`) en vez del rect a escala 0.72; el ROM quedará exacto en todos los videos.
2. **Desviación de la barra (cm)**: drift horizontal de `bar_path` con la calibración → cue de técnica en el panel.
3. **Peso estimado por color de disco** (GS Colour: gris5/verde10/amarillo15/azul20/rojo25 kg; disco 450mm,
   collar 50.4mm) → "Carga: X kg" tipo Lift App, una vez la detección sea fiable.
4. **Pérdida de velocidad por rep / fatiga** (% respecto a la mejor rep) en la tabla.
5. **Silueta aún más limpia**: entrenar `train_athlete_seg.py` (clase `athlete`) en Colab → `--segmentation-model`.
6. **Camino a móvil**: exportar modelos a ONNX/TFLite/CoreML (`model.export(format=...)`), medir tamaño/latencia,
   usar variantes `n` (nano). El objetivo final es app móvil.
7. **Optimización**: inferir a ~720p (`--max-resolution`) y mapear coords; warp de la máscara por flujo óptico
   en los frames saltados para silueta sin lag.

## Reglas finales para Codex
- Cambios pequeños y verificados; corre `pytest` y un smoke tras cada bloque; extrae frames y compáralos.
- No borres ni reescribas módulos que funcionan; respeta las decisiones de la sección 0.
- Si un cambio toca un contrato con test, actualiza el test (no lo borres por conveniencia).
- Memorias de contexto del proyecto: `docs/NEXT_STEPS.md` (roadmap), y los caveats de calibración y de la
  ficha de discos. PR de referencia con todo lo previo: https://github.com/juanrdzmb/PowerAI/pull/5.
- No subas videos, frames crudos (`datasets/all_frames/`, `datasets/training/frames`), caches ni pesos
  base a git (ya están en `.gitignore`). Sí commitea código, labels pequeñas y configs.
