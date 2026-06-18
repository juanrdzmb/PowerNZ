# Modelos De PowerAI

Esta carpeta es el punto de entrada de los modelos locales. Los pesos no se versionan en Git, pero la app espera estos nombres.

## Modelos Principales

| Archivo | Tipo | Clases esperadas | Uso |
|---|---|---|---|
| `powerai_bar_detector.pt` | YOLO detect | `plate`, `bar_hub` | Detecta el disco visible y el hub/manga de la barra. |
| `powerai_athlete_seg.pt` | YOLO segment | `athlete` | Segmenta la silueta del atleta. |
| `pose_landmarker_lite.task` | MediaPipe Tasks | pose | Backend alternativo de pose. |
| `yolo11s-seg.pt`, `yolo11n-seg.pt` | YOLO segment | `person` | Fallback si falta el modelo propio de atleta. |

## Como Los Descargo

La forma normal es:

```powershell
python model_downloader.py
```

O doble clic en `descargar_modelos.bat`.

El archivo `model_manifest.json` define las URLs, hashes y rutas locales. Ahora apunta a Hugging Face (`juanrdzmb/PowerNZ-Models`). Si cambio de cuenta o hosting, solo actualizo ese manifest.

## Como Carga La App

1. Si existe `models/powerai_bar_detector.pt`, lo uso automaticamente para `plate` y `bar_hub`.
2. Si existe `models/powerai_athlete_seg.pt`, lo uso automaticamente para la mascara del atleta.
3. Si el modelo de atleta no carga, intento modelos genericos de persona.
4. Si el detector de barra no carga, puedo caer a la heuristica de color. Con detector entrenado cargado, esa heuristica queda apagada salvo `--enable-plate-heuristic`.

## Contrato De Barra

El detector de barra debe ser un modelo de deteccion, no segmentacion.

Clases admitidas:

```text
0 = plate
1 = bar_hub
```

El orden real del `.pt` no me preocupa mientras los nombres sean correctos. El codigo compara por nombre.

## Contrato De Atleta

El modelo de atleta debe ser un modelo de segmentacion.

Clase principal:

```text
0 = athlete
```

Durante entrenamiento tambien puedo usar `background_person` como clase auxiliar/negativa, pero el runtime solo necesita `athlete`. Si el `.pt` no es `segment` o no expone `athlete/person`, se descarta y se intenta el siguiente fallback.

## Validar Los Pesos

Comprobacion rapida:

```powershell
@'
from ultralytics import YOLO
for path in ["models/powerai_bar_detector.pt", "models/powerai_athlete_seg.pt"]:
    model = YOLO(path)
    print(path, model.task, model.names)
'@ | python -
```

Resultado esperado:

```text
models/powerai_bar_detector.pt detect {0: 'plate', 1: 'bar_hub'}
models/powerai_athlete_seg.pt segment {0: 'athlete'}
```

## Publicar Modelos

No subo `.pt` al repo. Para esta v1 los publico en Hugging Face como `juanrdzmb/PowerNZ-Models`. Si necesito subir una nueva version, actualizo los pesos alli y cambio hashes/URLs en `model_manifest.json` cuando corresponda.
