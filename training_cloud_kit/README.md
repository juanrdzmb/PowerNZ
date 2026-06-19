# PowerNZ Cloud Training Kit

Este kit esta separado del codigo principal de PowerNZ. Sirve para preparar datos, entrenar modelos en Kaggle y validar los pesos descargados antes de integrarlos en la app.

## Que vamos a entrenar

**Un solo modelo de barra** (`PowerNZ_bar_detector.pt`) entrenado con frames de los **tres ejercicios** (peso muerto, sentadilla, press banca) y desde **multiples angulos**. Detecta dos clases: `plate` y `bar_hub`. Al ser un unico modelo multi-vista, sirve para los tres ejercicios; lo que cambia por ejercicio es la biomecanica en la app (flag `--exercise`), no el detector.

Modelos objetivo:

- `PowerNZ_bar_detector.pt`: obligatorio. Detecta `plate` y `bar_hub`.
- `PowerNZ_athlete_seg.pt`: opcional/futuro. Segmenta el atleta si etiquetas mascaras (mismos frames).

## Camino actual: Roboflow -> Kaggle (recomendado)

El etiquetado se hace en Roboflow (o Label Studio local) en un unico proyecto con clases `plate` y `bar_hub`. Cuando termines:

1. **Exporta desde Roboflow** en formato **YOLOv8** y descarga el ZIP.
2. **Convierte al layout del kit** (aplana `train/valid/test` y arregla el orden de clases, porque Roboflow suele exportar `Bar_hub`=0/`Plate`=1, invertido):

   ```powershell
   python training_cloud_kit\preparar_dataset_roboflow.py --roboflow-export "C:\ruta\al\export.zip"
   ```

   Genera `training_cloud_kit\work\PowerNZ_bar_v1_corrected.zip` (con `frames/` + `labels/` y clases `plate`=0, `bar_hub`=1). El script imprime el remapeo que aplico para que lo verifiques.
3. **Entrena en Kaggle** (ver "Entrenar en Kaggle paso a paso" abajo) con `MODE='train'`.
4. **Valida e integra** el modelo descargado (seccion 7).

> No hace falta `MODE='autolabel'`: ya tienes etiquetas manuales. Vas directo a `MODE='train'`.

### Entrenar en Kaggle paso a paso (si no lo has usado nunca)

1. Crea cuenta en kaggle.com y verificala (telefono) para poder usar GPU.
2. Arriba a la derecha: **Create > New Notebook**.
3. Panel derecho **Session options**: Accelerator = **GPU T4 x2** (o P100) e **Internet = On**.
4. **Add Input > Upload** (o *Datasets > New Dataset*): sube `PowerNZ_bar_v1_corrected.zip` como dataset. Quedara en `/kaggle/input/...`.
5. Sube tambien el script: arrastra `training_cloud_kit\kaggle_train_PowerNZ.py` al notebook, o abre `kaggle_train_PowerNZ.ipynb`.
6. En la primera celda instala dependencias y ejecuta el entrenamiento:

   ```python
   !pip -q install ultralytics
   !python kaggle_train_PowerNZ.py --mode train
   ```

   (El script descubre solo el ZIP en `/kaggle/input`. Si Kaggle lo muestra ya descomprimido, pasa `--dataset-root /kaggle/input/<nombre>`.)
7. Si se queda sin memoria, baja el batch: `--detector-batch 4`.
8. Al terminar, en el panel derecho **Output** descarga `PowerNZ_trained_models_v1.zip`.
9. En tu PC, valida e integra (seccion 7 mas abajo).

### Integrar el modelo de barra en la app

Copia `PowerNZ_bar_detector.pt` a `models\PowerNZ_bar_detector.pt`. La app lo detecta solo. Con el modelo entrenado presente conviene desactivar el heuristico de color para maxima precision:

```powershell
python main.py --input video.mp4 --output salida.mp4 --exercise squat --disable-plate-heuristic
```

Recuerda el flag `--exercise {deadlift,squat,bench}`: cambia la deteccion de repeticiones (peso muerto sube primero; sentadilla y press bajan primero) y las indicaciones de tecnica.

### Futuro: modelo de mascara del atleta (segmentacion)

Mismo circuito, con poligonos en vez de cajas:

1. Etiqueta la silueta del atleta (mascara) sobre **los mismos frames**, en un proyecto de **Instance Segmentation** con una clase `athlete`.
2. Exporta de Roboflow en **YOLOv8** y conviertelo:

   ```powershell
   python training_cloud_kit\preparar_dataset_roboflow.py --roboflow-export "C:\ruta\export_seg.zip" --task seg
   ```

   Genera `PowerNZ_athlete_v1_corrected.zip` con `frames/` + `masks/` (poligonos).
3. Lo mas simple: como `kaggle_train_PowerNZ.py` entrena detector y segmentacion en una sola pasada si el dataset tiene `labels/` **y** `masks/`, lo ideal es juntar las mascaras con el dataset de barra (mismos frames). Cuando llegues a este punto, pasame el export y preparo el merge; el resto del circuito (Kaggle, validacion) es identico.
4. Al integrar, copia `PowerNZ_athlete_seg.pt` a `models\PowerNZ_athlete_seg.pt`. La app lo usa automaticamente para la silueta (`segmentation.py` lo prioriza si existe).

## Flujo completo

### 0. Carpetas por ejercicio

Los videos manuales quedan organizados fuera del codigo principal, en:

```text
C:\Users\Juanda\Documents\Entrenamiento\
  Peso muerto\
  Sentadilla\
  Press Banca\
```

Puedes dejar los `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` o `.m4v` directamente dentro de la carpeta del ejercicio. Los scripts generan:

```text
frames\     imagenes para subir a Roboflow
labels\     etiquetas YOLO exportadas/copiadas despues
masks\      mascaras si entrenamos atleta mas adelante
manifest.csv
PowerNZ_<exercise>_v1_frames.zip
```

Para sacar frames:

```powershell
.\training_cloud_kit\extraer_frames_peso_muerto.ps1
.\training_cloud_kit\extraer_frames_sentadilla.ps1
.\training_cloud_kit\extraer_frames_press_banca.ps1
```

Cada script acepta videos con cualquier nombre y muestrea 1 frame cada ~0.5 s (stride 15), hasta 150 frames por video, a 1280 px maximo por lado. Asi salen frames variados y no cientos casi identicos seguidos, que es lo que conviene para etiquetar a mano. Si quieres mas o menos frames, ajusta `--stride` y `--max-frames-per-video` dentro del script.

Cada carpeta de ejercicio es a la vez la fuente de videos y la salida (`--work-root` apunta a la propia carpeta). Al volver a ejecutar con `--overwrite` solo se borran `frames\`, `labels\`, `masks\`, `review\` y los ZIP/manifests generados; los videos nunca se tocan.

### 1. Preparar frames locales

Desde la raiz del repo:

   ```powershell
   python training_cloud_kit\prepare_PowerNZ_cloud_dataset.py --stage frames --overwrite
   ```

Esto crea:

- `training_cloud_kit\work\deadlift_v1\PowerNZ_deadlift_v1_frames.zip`
- `training_cloud_kit\work\deadlift_v1\frames`
- `training_cloud_kit\work\deadlift_v1\manifest.csv`

### 2. Subir a Kaggle

1. En Kaggle, crea un dataset nuevo.
2. Sube `PowerNZ_deadlift_v1_frames.zip`.
3. Crea un notebook nuevo.
4. En Settings, activa GPU.
5. En Add Data, anade el dataset que acabas de subir. Tiene que contener `PowerNZ_deadlift_v1_frames.zip` o una carpeta con `frames/`.
6. Sube tambien `training_cloud_kit\kaggle_train_PowerNZ.py` o incluyelo en otro dataset de Kaggle. Este dataset del script no sustituye al dataset de frames.
7. Abre `training_cloud_kit\kaggle_train_PowerNZ.ipynb`.

### 3. Auto-etiquetar en Kaggle

En el notebook, deja:

```python
MODE = 'autolabel-clean'
DATASET_ZIP = ''
DATASET_ROOT = ''
```

Si Kaggle muestra el dataset como carpeta ya descomprimida, puedes poner la ruta en `DATASET_ROOT`, por ejemplo:

```python
DATASET_ROOT = '/kaggle/input/nombre-del-dataset/deadlift_v1'
```

Si dejas `DATASET_ZIP` y `DATASET_ROOT` vacios, el script buscara primero un `.zip` y luego una carpeta que tenga `frames/`.

Ejecuta las celdas. Al terminar tendras dos salidas:

- `/kaggle/working/PowerNZ_cloud_training/PowerNZ_autolabel_review.zip`
- `/kaggle/working/PowerNZ_deadlift_v1_corrected.zip`

El primer ZIP trae:

- `labels/`: etiquetas generadas.
- `previews/`: imagenes para revisar rapido si las cajas estan bien.

El segundo ZIP ya viene limpiado automaticamente en Kaggle. Ese es el recomendado para entrenar sin cargar tu PC.

### 4. Limpiar etiquetas automaticamente en local

Este paso solo hace falta si quieres limpiar en tu PC. Si ya tienes `PowerNZ_deadlift_v1_corrected.zip` desde Kaggle, saltalo.

Descarga `PowerNZ_autolabel_review.zip` desde Kaggle y dejalo en Descargas. Luego ejecuta:

```powershell
python training_cloud_kit\auto_clean_autolabels.py --prepare-frames-if-missing
```

Esto hace automaticamente:

- encuentra el ZIP descargado;
- copia las etiquetas generadas por Kaggle;
- se queda con la mejor pareja `plate` + `bar_hub` por frame;
- descarta cajas grandes, raras o sin pareja coherente;
- crea previews limpias en `training_cloud_kit\work\deadlift_v1\review\auto_cleaned_previews`;
- genera `training_cloud_kit\work\deadlift_v1\PowerNZ_deadlift_v1_corrected.zip`.

Ese ZIP corregido es el que debes volver a subir a Kaggle para entrenar.

### 4b. Etiquetado manual en local con Label Studio (recomendado)

Todo el etiquetado se hace en tu PC, nada se sube a internet (al contrario que Roboflow, que es publico). Label Studio sirve las imagenes desde `C:\Users\Juanda\Documents\Entrenamiento` y guarda los proyectos y anotaciones en `training_cloud_kit\label_studio_data` (ignorado por git).

#### Arrancar

```powershell
.\training_cloud_kit\lanzar_label_studio.ps1
```

El script, la primera vez, crea el entorno virtual e instala Label Studio. Despues siempre:

- arranca el servidor en `http://localhost:8080`;
- crea (o reutiliza) un proyecto por ejercicio: `PowerNZ - Peso muerto`, `PowerNZ - Sentadilla`, `PowerNZ - Press Banca`;
- sincroniza de golpe todas las imagenes de cada carpeta `frames\`;
- abre el navegador.

Para parar el servidor: `Ctrl+C` en esa ventana. Los proyectos y lo etiquetado quedan guardados para la proxima vez.

Usuario y contrasena por defecto (solo local): `juanda@PowerNZ.local` / `PowerNZ-local-2026`.

Si extraes mas frames despues, vuelve a ejecutar para re-sincronizar (no duplica nada):

```powershell
.\training_cloud_kit\.venv_labelstudio\Scripts\python.exe training_cloud_kit\configurar_label_studio.py
```

> Por que asi y no arrastrando archivos: Label Studio no sirve archivos locales por defecto, y `file://` o trucos de CORS los bloquea el navegador. El metodo correcto para miles de imagenes es el almacenamiento local (`LOCAL_FILES_SERVING_ENABLED` + document root), que es justo lo que monta el lanzador.

#### Como etiquetar cada frame

Herramienta: caja rectangular (`RectangleLabels`). Atajos: `1` = plate, `2` = bar_hub. Dibuja la caja y, si quieres, pulsa Enter / boton Submit para pasar al siguiente.

Que marcar:

- `plate`: el disco cargado en la barra del atleta. Una caja por disco visible de esa barra. Ajusta la caja al borde del disco.
- `bar_hub`: el centro/sleeve (la zona donde entran los discos) de esa misma barra.

Que NO marcar (clave para no meter falsos positivos):

- discos en el suelo o apoyados en racks;
- discos o barras de fondo, de otra persona o de otra estacion;
- cualquier cosa redonda que no sea el disco cargado (ruedas, focos, etc.).

Buenas practicas:

- Si en un frame no hay barra cargada visible, dejalo sin cajas y pulsa Submit: los frames negativos ayudan al modelo.
- Mejor pocas cajas bien ajustadas que muchas dudosas.
- Mantén el criterio entre vistas lateral, diagonal y frontal.

#### Exportar a YOLO

Cuando termines un ejercicio: en el proyecto, `Export` -> formato **YOLO** -> descarga el ZIP. Dentro trae `labels\` (txt YOLO) y `classes.txt`.

Comprueba que `classes.txt` es exactamente:

```text
plate
bar_hub
```

Asi el indice 0 = plate y 1 = bar_hub, que es lo que esperan `dataset_bar.yaml` y el entrenamiento. Copia los `*.txt` del export a la carpeta `labels\` del ejercicio (`C:\Users\Juanda\Documents\Entrenamiento\<Ejercicio>\labels`) y sigue en el paso 5 para reempaquetar.

### 5. Reempaquetar dataset corregido

Si hiciste correcciones manuales despues de la limpieza automatica:

```powershell
python training_cloud_kit\prepare_PowerNZ_cloud_dataset.py --stage package
```

Esto crea:

- `training_cloud_kit\work\deadlift_v1\PowerNZ_deadlift_v1_corrected.zip`

### 6. Entrenar en Kaggle

Sube `PowerNZ_deadlift_v1_corrected.zip` a Kaggle como nuevo dataset o nueva version del dataset anterior. Si usaste `MODE = 'autolabel-clean'`, descarga ese ZIP directamente desde Kaggle y vuelve a subirlo como dataset de entrenamiento.

En el notebook, cambia:

```python
MODE = 'train'
DATASET_ZIP = ''
DATASET_ROOT = ''
```

Ejecuta las celdas. Si Kaggle se queda sin memoria, baja el batch en el comando del notebook o ejecuta el script asi:

```python
cmd += ['--detector-batch', '4']
```

Al terminar descarga:

- `/kaggle/working/PowerNZ_trained_models_v1.zip`

### 7. Validar modelos descargados

Sin instalar nada en la app, valida el ZIP:

   ```powershell
   python training_cloud_kit\validate_downloaded_models.py --models-zip path\to\PowerNZ_trained_models_v1.zip
   ```

Si pasa, entonces ya podemos integrar manualmente:

- `PowerNZ_bar_detector.pt` a `models\PowerNZ_bar_detector.pt`
- `PowerNZ_athlete_seg.pt` a `models\PowerNZ_athlete_seg.pt`, solo si existe

## Reglas para evitar falsos positivos

- Etiquetar solo el disco cargado en la barra y el hub/sleeve de esa barra.
- No etiquetar discos del suelo, discos de fondo ni objetos parecidos.
- Incluir frames negativos cuando aparezcan discos fuera de la barra.
- Revisar especialmente vistas laterales, diagonales, frontales y videos con mas de una persona.
- Preferir menos etiquetas pero bien corregidas antes que muchas cajas dudosas.

## Estructura generada

El kit crea archivos grandes dentro de `training_cloud_kit\work`, que esta ignorado por git.

```text
training_cloud_kit/
  work/
    deadlift_v1/
      frames/
      labels/
      masks/
      dataset_bar.yaml
      dataset_athlete.yaml
      manifest.csv
      PowerNZ_deadlift_v1_frames.zip
      PowerNZ_deadlift_v1_corrected.zip
```

## Notas de Kaggle

Kaggle es la via principal porque el entrenamiento necesita GPU. Colab queda como alternativa si Kaggle falla.

Comandos utiles dentro de Kaggle, si prefieres no usar el notebook:

```bash
python kaggle_train_PowerNZ.py --mode autolabel
python kaggle_train_PowerNZ.py --mode train
```

El ZIP final siempre se genera en:

```text
/kaggle/working/PowerNZ_trained_models_v1.zip
```
