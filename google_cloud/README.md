# PowerNZ: mejora de tus modelos en Google Cloud

Esta guía está pensada para seguirla literalmente. No presupone experiencia con Roboflow, GPUs ni Google Cloud.

## Qué pasa con tus modelos actuales

No se descartan ni se vuelve a empezar de cero. Si existen estos archivos, el script los sube de forma privada junto al dataset y continúa su entrenamiento (**fine-tuning**):

- `models\powerai_bar_detector.pt`: tu detector de discos y centro/eje de barra;
- `models\powerai_athlete_seg.pt`: tu segmentador del atleta.

El detector conserva las características visuales que ya aprendió de tus barras y discos. Si el nuevo dataset añade `loose_plate` y `rack_plate`, se adapta la salida a las nuevas clases, pero se reutiliza el aprendizaje visual anterior. El segmentador hace lo mismo con tus máscaras: parte de tu clase `athlete` y aprende además `background_person` si la incluyes.

El resultado nuevo se guarda primero en `outputs\google_cloud\<job>`. Antes de instalarlo, el instalador conserva una copia exacta del modelo activo en `models\backups\`. Por tanto, siempre puedes volver al anterior. Tus datasets y máscaras de Roboflow tampoco se modifican.

Solo se usa un peso genérico cuando no se encuentra tu modelo actual, o cuando lo pides expresamente con, por ejemplo, `-BaseModel yolo26s.pt`. También puedes indicar cualquier peso local con `-BaseModel "C:\ruta\modelo.pt"`.

## Resumen: qué entrenar y en qué orden

No entrenes los cuatro modelos a la vez. Hazlos en este orden:

1. **Detector v2 (`detect`)**: `plate`, `bar_hub`, `loose_plate`, `rack_plate`. Es el que más mejorará la trayectoria y los falsos positivos ahora mismo. La aplicación usa `plate` y `bar_hub`; las otras dos clases enseñan al modelo qué debe ignorar.
2. **Segmentación v2 (`segment`)**: `athlete`, `background_person`. Evita que máscara y pose salten a spotters o gente del gimnasio.
3. **Keypoints de barra (`pose`)**: una clase `barbell` y cinco puntos. Dará centros más precisos, pero el peso necesitará una integración adicional en el pipeline después de entrenarlo.
4. **Barra orientada (`obb`)**: opcional. Aporta el ángulo de la barra en frontal/diagonal, pero no lo entrenes antes de que detector y máscara sean fiables.

Con tus créditos, empieza solo por `detect`. Comprueba el resultado en vídeos reales y después haz `segment`.

## Qué máquina se usa

El script crea:

- zona: `europe-west4-a` (Países Bajos);
- máquina: `n1-standard-8` (8 vCPU, 30 GB RAM);
- GPU: una NVIDIA T4 de 16 GB;
- disco: 100 GB `pd-balanced`;
- imagen: PyTorch 2.9 + CUDA 12.9 de Deep Learning VM;
- duración máxima de entrenamiento: 8 horas;
- apagado automático al terminar o fallar.

Una T4 es suficiente para YOLO26s con tamaño 768/960 y batch automático. Google publica la T4 a partir de 0,35 USD/h solo por GPU; CPU, RAM y disco se cobran aparte. Consulta siempre la estimación mostrada en tu consola porque región, moneda y modalidad cambian el total: <https://cloud.google.com/products/compute/gpus-pricing>.

El modo normal es más fiable. Para ahorrar puedes añadir `-Spot`, pero Google puede detener una VM Spot en cualquier momento. No lo uses en el primer entrenamiento.

## Parte A — Preparar los fotogramas

### A1. Cantidades recomendadas

Para el detector v2, objetivo inicial:

- 3.000–5.000 fotogramas únicos;
- aproximadamente un tercio por ejercicio: peso muerto, sentadilla y banca;
- 35 % lateral, 35 % diagonal, 20 % frontal y 10 % ángulos difíciles;
- 25–35 % de fotogramas negativos o difíciles;
- personas, gimnasios, discos, iluminación y cámaras diferentes.

Para segmentación:

- 1.500–2.500 máscaras bien revisadas;
- al menos 40 % con varias personas o spotters;
- incluye ropa oscura, atleta parcialmente oculto, banco y discos delante del cuerpo.

Para keypoints de barra:

- 2.500–4.000 fotogramas;
- mucha variedad de inclinación, oclusión y distancia;
- no empieces este proyecto hasta terminar el detector v2.

### A2. Qué momentos elegir

Incluye siempre:

- barra en reposo antes de empezar;
- inicio, mitad y final de cada fase;
- punto más bajo y bloqueo;
- discos parcialmente fuera de pantalla;
- movimiento rápido con desenfoque;
- contraluz, vídeo comprimido y poca luz;
- otra barra o discos en el fondo;
- discos sueltos en suelo/rack;
- más de una persona;
- frames sin atleta o sin barra cargada.

No uses 50 fotogramas casi idénticos seguidos. Muestrea normalmente uno cada 0,5–1 segundo y añade manualmente frames alrededor del fondo/bloqueo. Los scripts existentes de `training_cloud_kit` ya extraen frames aproximadamente cada 0,5 segundos.

### A3. Regla crítica para train/valid/test

Todos los fotogramas de un mismo vídeo deben quedarse en el mismo split. Nunca pongas frames del mismo clip en entrenamiento y validación: el resultado parecerá excelente aunque el modelo no generalice.

Reparto recomendado por vídeo:

- 70 % de vídeos para `train`;
- 20 % para `valid`;
- 10 % para `test`.

## Parte B — Roboflow, clic por clic

Roboflow permite crear una versión y descargarla como ZIP en formatos YOLO: <https://docs.roboflow.com/datasets/download-a-dataset>.

### B1. Proyecto del detector v2

1. Entra en <https://app.roboflow.com/> y crea una cuenta.
2. Crea un Workspace privado si tus vídeos no deben ser públicos.
3. Pulsa **Create New Project**.
4. Tipo: **Object Detection**.
5. Nombre: `PowerNZ Bar Detector v2`.
6. Crea exactamente estas clases, respetando minúsculas:

   ```text
   plate
   bar_hub
   loose_plate
   rack_plate
   ```

7. Sube los fotogramas extraídos.
8. No subas vídeos personales a un workspace público.

Cómo etiquetar:

- `plate`: solo el disco cargado en la barra que usa el atleta;
- `bar_hub`: centro/sleeve de esa misma barra;
- `loose_plate`: disco en el suelo o apoyado fuera de una barra activa;
- `rack_plate`: disco guardado en un rack;
- una polea, rueda, foco u objeto circular falso se deja sin etiqueta: ese frame funciona como negativo;
- si no hay ninguna barra válida, guarda el frame sin `plate` ni `bar_hub`.

Las cajas deben abrazar el objeto. No dibujes una caja enorme alrededor de toda la barra para `bar_hub`.

### B2. Proyecto de segmentación

1. Crea otro proyecto.
2. Tipo: **Instance Segmentation**.
3. Nombre: `PowerNZ Athlete Seg v2`.
4. Clases exactas:

   ```text
   athlete
   background_person
   ```

5. `athlete` es quien realiza el levantamiento.
6. Spotters, entrenadores y personas del fondo son `background_person`.
7. El polígono debe seguir la silueta; no incluyas barra, banco ni discos.

SAM 2 puede ayudarte a propagar una máscara a través de un vídeo, pero revisa cada secuencia: <https://ai.meta.com/research/sam2/>.

### B3. Proyecto de keypoints de barra (más adelante)

Tipo: **Keypoint Detection**. Clase: `barbell`.

Orden fijo de puntos:

1. `left_plate_center` — centro del disco situado a la izquierda de la imagen;
2. `right_plate_center`;
3. `left_sleeve`;
4. `right_sleeve`;
5. `bar_center`.

Si un punto está oculto, márcalo como no visible; no lo inventes. El modelo pose devuelve cajas y puntos normalizados y Ultralytics admite datasets personalizados de keypoints: <https://docs.ultralytics.com/datasets/pose/>.

### B4. Generar y descargar la versión

1. Revisa al menos 100 imágenes al azar.
2. Abre **Versions** → **Generate New Version**.
3. Preprocesado recomendado:
   - Auto-Orient: activado;
   - Resize: `Fit within 1280x1280`;
   - no recortar al atleta ni la barra.
4. No añadas aumentos fuertes en Roboflow; el entrenador ya aplica brillo, escala, traslación, mosaic y mixup.
5. Comprueba que los splits están separados por vídeo.
6. Pulsa **Download Dataset**.
7. Elige **YOLO26** o **YOLOv8**. Ambos incluyen TXT y `data.yaml` y sirven para este kit.
8. Selecciona **Download ZIP to computer**.
9. Guarda el ZIP, por ejemplo:

   ```text
   C:\Users\Juanda\Downloads\PowerNZ-Bar-v2-YOLO26.zip
   ```

### B5. Datasets públicos útiles

El único que recomiendo como arranque claro es **Barbell Annotation**, 7.867 imágenes, clase `barbell`, licencia CC BY 4.0: <https://universe.roboflow.com/barbell-annotation/barbell-8nrjn>.

Úsalo solo para diversidad visual, no como dataset final:

- no tiene tus clases `plate`/`bar_hub`;
- revisa y reetiqueta las imágenes que importes;
- limita datos públicos genéricos a 10–20 % del total;
- conserva la atribución exigida por CC BY 4.0;
- no mezcles un dataset cuya licencia no esté indicada.

Roboflow Universe permite clonar imágenes seleccionadas o descargar un ZIP: <https://docs.roboflow.com/universe/download-a-universe-dataset>.

No recomiendo usar un modelo público directamente: sus clases y cámaras no coinciden con PowerNZ. Tu modelo actual afinado con negativos propios será más fiable.

## Parte C — Preparar Google Cloud una sola vez

### C1. Crear/seleccionar proyecto y facturación

1. Entra en <https://console.cloud.google.com/>.
2. Arriba, abre el selector de proyecto.
3. Crea un proyecto; ejemplo: `powernz-training`.
4. Copia el **Project ID**, no solo el nombre visible. Ejemplo:

   ```text
   powernz-training-123456
   ```

5. Vincula la cuenta de facturación que contiene tus créditos.
6. Ve a **Billing → Budgets & alerts** y crea un presupuesto de 50 € con alertas al 50 %, 80 % y 100 %.

Importante: una alerta de presupuesto avisa, pero no detiene automáticamente el gasto: <https://cloud.google.com/billing/docs/how-to/budgets>.

### C2. Solicitar cuota de GPU

Los proyectos nuevos suelen tener cuota GPU igual a cero.

1. Abre **IAM & Admin → Quotas & System Limits**.
2. Filtra por servicio **Compute Engine API**.
3. Busca cuota global **GPUs (all regions)** y solicita `1`.
4. Busca la cuota regional de **NVIDIA T4 GPUs** en `europe-west4` y solicita `1`.
5. Si vas a usar `-Spot`, revisa también la cuota de GPUs preemptibles/Spot.
6. Espera la aprobación antes de ejecutar el entrenamiento.

Google exige cuota global y cuota del tipo de GPU en la región: <https://cloud.google.com/compute/resource-usage>.

### C3. Instalar Google Cloud CLI en Windows

1. Sigue el instalador oficial: <https://cloud.google.com/sdk/docs/install-sdk>.
2. Acepta instalar el Python incluido y añadir `gcloud` al PATH.
3. Cierra y abre PowerShell.
4. Ejecuta:

   ```powershell
   gcloud init
   ```

5. Se abrirá el navegador. Inicia sesión con la cuenta de Google Cloud.
6. Selecciona el proyecto creado.
7. Comprueba:

   ```powershell
   gcloud auth list
   gcloud config list
   ```

No descargues ni guardes claves JSON. El kit crea una cuenta de servicio ligada a la VM sin claves privadas locales.

## Parte D — Entrenar con un solo comando

Abre PowerShell dentro de la carpeta PowerNZ:

```powershell
cd C:\Users\Juanda\Documents\PowerNZ
```

Instala las dependencias locales si todavía no lo hiciste:

```powershell
python -m pip install -r requirements.txt
```

Permite los scripts solo para esta ventana:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### D1. Detector v2

Sustituye el Project ID y la ruta del ZIP:

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 `
  -ProjectId "powernz-training-123456" `
  -DatasetZip "C:\Users\Juanda\Downloads\PowerNZ-Bar-v2-YOLO26.zip" `
  -Task detect
```

El script hará automáticamente:

1. comprobar cuenta/proyecto;
2. abrir el ZIP y validar `data.yaml`, splits y clases antes de gastar dinero;
3. habilitar Compute, Storage e IAM;
4. crear un bucket privado;
5. crear una cuenta de servicio sin claves;
6. subir ZIP, entrenador y una copia privada de tu modelo actual;
7. crear la VM T4;
8. esperar mientras entrena;
9. descargar `best.pt` y métricas;
10. validar tarea y clases;
11. guardar copia del modelo anterior;
12. instalar `models\powerai_bar_detector.pt`;
13. actualizar SHA256 en `models\model_manifest.json`;
14. borrar la VM y su disco.

Puedes cerrar la ventana después de crear la VM si añades `-NoWait`. La VM se apaga sola. El comando imprimirá cómo recoger el resultado después.

### D2. Segmentación

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 `
  -ProjectId "powernz-training-123456" `
  -DatasetZip "C:\Users\Juanda\Downloads\PowerNZ-Athlete-v2-YOLO26.zip" `
  -Task segment
```

Instalará `models\powerai_athlete_seg.pt`.

Por defecto continuará desde tu `models\powerai_athlete_seg.pt`; no pierde la segmentación que dibujaste manualmente.

### D3. Keypoints u OBB

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 -ProjectId "TU_PROJECT_ID" -DatasetZip "C:\ruta\bar-pose.zip" -Task pose
.\google_cloud\New-PowerNZTrainingJob.ps1 -ProjectId "TU_PROJECT_ID" -DatasetZip "C:\ruta\bar-obb.zip" -Task obb
```

Estos pesos quedan en `models\`, pero todavía necesitan adaptar el pipeline. Entrénalos solo cuando podamos revisar juntos sus clases/keypoints y métricas.

### D4. Opciones útiles

Entrenamiento barato pero interrumpible:

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 ... -Task detect -Spot
```

Reducir memoria si aparece CUDA out of memory:

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 ... -Task detect -Batch 6
```

Probar rápido antes del entrenamiento final:

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 ... -Task detect -Epochs 5 -ImageSize 640
```

Conservar la VM detenida para inspeccionarla (seguirá cobrando disco):

```powershell
.\google_cloud\New-PowerNZTrainingJob.ps1 ... -Task detect -KeepVm
```

## Parte E — Recoger un entrenamiento lanzado con `-NoWait`

El primer script muestra el JobId, InstanceName y el comando exacto. Será parecido a:

```powershell
.\google_cloud\Get-PowerNZTrainingResult.ps1 `
  -ProjectId "powernz-training-123456" `
  -JobId "powernz-detect-20260702-150000" `
  -Task detect `
  -Zone "europe-west4-a" `
  -InstanceName "powernz-detect-20260702-150000" `
  -Wait
```

Los informes quedan en:

```text
outputs\google_cloud\<job-id>\
  powerai_bar_detector.pt
  training_summary.json
  training_artifacts.tar.gz
  startup.log
```

## Parte F — Probar y publicar el nuevo modelo

Después de instalar detector/segmentador:

```powershell
python model_downloader.py --check
python -m pytest -q
```

Analiza al menos un vídeo reservado de cada ejercicio y ángulo. No uses vídeos que estuvieron en entrenamiento.

Si el resultado mejora, publica solo el peso nuevo en Hugging Face:

```powershell
hf auth login
python upload_models_to_huggingface.py --only powerai_bar_detector.pt
```

Para segmentación:

```powershell
python upload_models_to_huggingface.py --only powerai_athlete_seg.pt
```

El instalador actualiza el hash del manifest antes de subir. Después haz commit de `models/model_manifest.json`; los `.pt` siguen ignorados por Git.

## Cómo decidir si el modelo es mejor

No mires solo mAP. Guarda un conjunto de 20–30 vídeos nunca usados y mide:

- falsos positivos por minuto;
- porcentaje de frames con hub/disco correcto;
- error del centro respecto al centro real del disco;
- número de cortes y saltos de trayectoria;
- error de repeticiones por vídeo;
- máscara sobre el atleta correcto;
- rendimiento separado por lateral, diagonal y frontal.

No sustituyas el modelo anterior si mejora la media pero empeora claramente una vista o un ejercicio.

## Solución de problemas

### `Quota 'NVIDIA_T4_GPUS' exceeded`

Falta cuota T4 regional o cuota global. Vuelve a la Parte C2.

### `ZONE_RESOURCE_POOL_EXHAUSTED`

No hay una T4 libre en esa zona. Repite con `-Zone "europe-west4-b"` o `-Zone "europe-west1-b"`.

### `CUDA is not available`

La VM se detendrá y guardará `startup.log`. No reintentes muchas veces: revisa la imagen/GPU y pásame el log.

### `CUDA out of memory`

Repite con `-Batch 6`; si continúa, `-Batch 4` o `-ImageSize 768`.

### La ventana se cerró

La VM continúa y se apaga al finalizar. Usa el comando de la Parte E.

### El entrenamiento falló

El recolector descarga `startup.log` y `_FAILED`. La VM se borra salvo que uses `-KeepVm`. Comparte ese log, nunca credenciales ni tokens.

### Control de gasto

En Compute Engine → VM instances no debe quedar ninguna VM `powernz-*` en estado RUNNING después de terminar. Una VM STOPPED no cobra CPU/GPU, pero su disco sí; bórrala si ya descargaste el resultado.
