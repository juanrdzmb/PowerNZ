# PowerAI

PowerAI es mi herramienta para analizar videos de powerlifting con vision por computador. La estoy preparando como una primera version estable: detecto el atleta, la barra y los discos, cuento repeticiones logicas para peso muerto, sentadilla y press banca, y exporto un video vertical 9:16 listo para revisar o compartir.

El objetivo de esta v1 no es llenar la pantalla de lineas. Quiero que el video diga lo importante: donde esta el plato, donde esta el hub de la barra, como se mueve la barra, que velocidad lleva y si la repeticion cumple una lectura tecnica razonable basada en reglas IPF.

## Que Hace

- Usa `models/powerai_bar_detector.pt` para detectar `plate` y `bar_hub`.
- Usa `models/powerai_athlete_seg.pt` para segmentar la silueta del atleta.
- Usa YOLO Pose o MediaPipe para obtener landmarks corporales.
- Refina los landmarks con la mascara del atleta para evitar puntos saltando al fondo.
- Calcula velocidad de barra, ROM, drift horizontal, fases del levantamiento y repeticiones.
- Valida repeticiones con una aproximacion 2D a criterios IPF:
  - peso muerto: extension de rodilla/cadera al bloqueo;
  - sentadilla: profundidad por debajo de paralela y bloqueo arriba;
  - press banca: recorrido suficiente y codos bloqueados arriba.
- Exporta por defecto en `720x1280` (`portrait-720`) sin recortar el video original.
- Genera reportes JSON/CSV si se piden.

## Modelos

Dejo los pesos descargados dentro de `models/`, pero no los subo a Git porque son grandes. El repo trae un descargador para bajarlos desde Hugging Face.

| Archivo | Uso |
|---|---|
| `models/powerai_bar_detector.pt` | Detector entrenado de `plate` y `bar_hub`. Es la fuente principal de la barra. |
| `models/powerai_athlete_seg.pt` | Segmentacion entrenada del atleta (`athlete`). Es la mascara principal. |
| `yolov8n-pose.pt` | Pose YOLO incluida para landmarks. |
| `models/yolo11s-seg.pt`, `models/yolo11n-seg.pt` | Fallback de segmentacion de persona si falta el modelo propio. |

En v1 la heuristica por color de discos esta apagada cuando el detector entrenado carga bien. Si quiero forzar el respaldo por color uso `--enable-plate-heuristic`.

Para descargar los modelos entrenados:

```powershell
python model_downloader.py
```

Tambien puedo hacer doble clic en `descargar_modelos.bat`.

Las URLs viven en `models/model_manifest.json`. Para esta v1 apuntan a `juanrdzmb/PowerNZ-Models` en Hugging Face. Si cambio de cuenta o hosting, solo actualizo ese manifest.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python model_downloader.py
```

## Uso Rapido

La forma mas comoda en Windows es abrir:

```text
abrir_powernz.bat
```

Desde esa ventana elijo el video, el ejercicio, la salida y pulso `Analizar video`. Si faltan modelos, la app me pregunta si quiero descargarlos antes de empezar.

Si quiero usar la terminal, tambien puedo ejecutar los comandos directamente.

Peso muerto:

```powershell
python main.py --input video_prueba.mp4 --output outputs\deadlift.mp4 --exercise deadlift --pose-backend yolo --plate-diameter-px 120
```

Sentadilla:

```powershell
python main.py --input "C:\Users\Juanda\Documents\Entrenamiento\Sentadilla\Sentadilla  (2).mp4" --output outputs\squat.mp4 --exercise squat --pose-backend yolo --plate-diameter-px 120
```

Press banca:

```powershell
python main.py --input "C:\Users\Juanda\Documents\Entrenamiento\Press Banca\PressBanca (4).mp4" --output outputs\bench.mp4 --exercise bench --pose-backend yolo --plate-diameter-px 120
```

Por defecto el resultado sale en `720x1280` con fondo discreto si el video original no es vertical. No recorto al atleta ni la barra.

## Opciones Importantes

- `--exercise deadlift|squat|bench`: cambia la maquina de estados y las reglas tecnicas.
- `--output-format portrait-720`: default; exporta 9:16 `720x1280` sin recortar.
- `--output-format source`: mantiene la geometria original, util para depurar.
- `--object-model ruta.pt`: usa otro detector `plate/bar_hub`.
- `--enable-plate-heuristic`: activa el respaldo antiguo por color.
- `--disable-trained-object-model`: ignora el detector entrenado.
- `--segmentation-backend auto`: usa primero `powerai_athlete_seg.pt`, despues fallback.
- `--segmentation-model ruta.pt`: usa otro modelo de segmentacion.
- `--velocity-chart bar`: default; grafico inferior solo de barra.
- `--velocity-chart multi`: muestra barra + landmarks en el grafico para depurar.
- `--body-velocity-display compact`: muestra velocidades corporales como indicadores pequeños.
- `--report-json outputs\report.json`: guarda resumen tecnico.
- `--report-csv outputs\reps.csv`: guarda datos por repeticion.
- `--validation-run-label nombre --save-validation-screenshots`: guarda video, reportes y capturas de revision.

## Flujo Interno

Proceso el video en dos pasadas.

1. En la primera pasada analizo: pose, mascara, barra, tracking, velocidad y repeticiones.
2. Cuando termina esa pasada ya conozco el total de reps, la curva completa y la escala estable del grafico.
3. En la segunda pasada dibujo el overlay final con contador `hecho/total`, trayectoria y tabla.

El punto metrico de la barra es `bar_hub`. Si veo el plato pero no veo un hub fiable, dibujo la caja del plato para que se entienda el seguimiento, pero no invento velocidad ni trayectoria desde la pose.

## Validacion Visual

Antes de considerar una version lista reviso tres cosas:

- El rectangulo `Plate` abraza el disco detectado y lo sigue sin saltar a discos del suelo.
- La caja `Bar` aparece solo cuando el `bar_hub` es fiable.
- La mascara y el esqueleto pertenecen al atleta que sostiene la barra.

Comando recomendado de smoke:

```powershell
python main.py --input "C:\Users\Juanda\Documents\Entrenamiento\Peso muerto\Peso Muerto (1).mp4" --output outputs\v1_deadlift.mp4 --exercise deadlift --pose-backend yolo --plate-diameter-px 120 --validation-run-label v1_deadlift --save-validation-screenshots
```

## Tests

```powershell
python -m pytest
```

La suite cubre geometria 720x1280, mascara del atleta aplicada a landmarks, segmentacion por modelo propio, flujo estricto sin heuristica de color, conteo por ejercicio, overlay y reportes.

## Notas De Version

Esta v1 esta pensada como base limpia para subir a GitHub. No versiono pesos `.pt`, videos, `outputs/`, `runs/` ni datasets locales. Para compartir modelos usare Releases, Hugging Face, Drive u otro almacenamiento externo y dejare los nombres esperados documentados.
