# Entrenar PowerNZ En RunPod

Esta guia es mi flujo limpio para entrenar modelos de PowerNZ en RunPod sin depender de Kaggle. La idea es alquilar una GPU por horas, subir el dataset, entrenar, bajar el `.zip` final y apagar el Pod.

## Que GPU Alquilo

Para detectar `plate` y `bar_hub`:

- Barato/equilibrado: RTX 3090, RTX 4090, A4000, A5000.
- Mas rapido si el dataset es grande: A6000, L40, L40S, A100.

Yo empezaria con **RTX 4090** si esta disponible. Si no, **RTX 3090** suele ir bien. Para entrenar segmentacion del atleta junto al detector, mejor 24 GB de VRAM o mas.

## Crear El Pod

1. Entro a RunPod.
2. Voy a `Pods`.
3. Creo un Pod con plantilla **RunPod PyTorch**.
4. Elijo una GPU de 24 GB o mas si puedo.
5. Disco recomendado:
   - Container Disk: 30-50 GB.
   - Volume o Network Volume: 50-100 GB si voy a entrenar varios datasets.
6. Activo JupyterLab y SSH si RunPod lo ofrece en la plantilla.
7. Arranco el Pod.

RunPod documenta Pods con GPU bajo demanda, plantillas PyTorch listas, conexion por SSH/Jupyter/VS Code y almacenamiento persistente o network volumes. Referencias oficiales:

- https://docs.runpod.io/pods/overview
- https://docs.runpod.io/pods/configuration/use-ssh
- https://docs.runpod.io/storage/network-volumes

## Entrar Al Pod

Puedo usar JupyterLab desde el boton `Connect`, o SSH. Para SSH, RunPod recomienda usar clave SSH:

```bash
ssh root@IP_DEL_POD -p PUERTO -i ~/.ssh/id_ed25519
```

En Windows tambien puedo usar PowerShell con el comando que copia RunPod en la pestaña `Connect`.

## Preparar PowerNZ

Dentro del Pod:

```bash
cd /workspace
git clone https://github.com/juanrdzmb/PowerNZ.git
cd PowerNZ
bash runpod/setup_runpod.sh
```

Si el repo ya existe:

```bash
cd /workspace/PowerNZ
git pull
bash runpod/setup_runpod.sh
```

## Subir Dataset

La forma mas comoda:

1. Exporto desde Roboflow en formato YOLOv8/YOLOv11.
2. Subo el `.zip` al Pod a:

```text
/workspace/datasets/
```

Opciones para subir:

- JupyterLab: panel izquierdo > Upload.
- SCP desde mi PC:

```powershell
scp -P PUERTO "C:\ruta\dataset.zip" root@IP_DEL_POD:/workspace/datasets/
```

## Si El Dataset Viene De Roboflow

Primero convierto el export al formato que espera el entrenador. Para detector de barra/discos:

```bash
cd /workspace/PowerNZ
python training_cloud_kit/preparar_dataset_roboflow.py \
  --roboflow-export /workspace/datasets/mi_export_roboflow.zip \
  --task detect \
  --dataset-name powernz_bar_runpod
```

El ZIP corregido queda normalmente en:

```text
training_cloud_kit/work/powernz_bar_runpod_corrected.zip
```

Para segmentacion del atleta:

```bash
python training_cloud_kit/preparar_dataset_roboflow.py \
  --roboflow-export /workspace/datasets/mi_export_seg.zip \
  --task seg \
  --dataset-name powernz_athlete_runpod
```

## Entrenar Detector Plate + Bar_hub

Comando recomendado:

```bash
cd /workspace/PowerNZ
bash runpod/train_runpod.sh \
  --dataset-zip training_cloud_kit/work/powernz_bar_runpod_corrected.zip \
  --detector-base-model yolo11s.pt \
  --detector-epochs 140 \
  --detector-imgsz 960 \
  --detector-batch 12 \
  --skip-athlete-seg
```

Si falta memoria:

```bash
bash runpod/train_runpod.sh \
  --dataset-zip training_cloud_kit/work/powernz_bar_runpod_corrected.zip \
  --detector-batch 8 \
  --skip-athlete-seg
```

Si quiero maxima calidad y la GPU aguanta:

```bash
bash runpod/train_runpod.sh \
  --dataset-zip training_cloud_kit/work/powernz_bar_runpod_corrected.zip \
  --detector-base-model yolo11m.pt \
  --detector-epochs 180 \
  --detector-imgsz 1024 \
  --detector-batch 8 \
  --skip-athlete-seg
```

## Entrenar Detector + Mascara Del Atleta

El script entrena segmentacion si el dataset contiene una carpeta `masks/`. Si solo tiene `labels/`, entrena detector.

```bash
bash runpod/train_runpod.sh \
  --dataset-zip /workspace/datasets/PowerNZ_dataset_con_labels_y_masks.zip \
  --detector-base-model yolo11s.pt \
  --detector-epochs 140 \
  --detector-imgsz 960 \
  --detector-batch 10 \
  --athlete-base-model yolo11s-seg.pt \
  --athlete-epochs 120 \
  --athlete-imgsz 768 \
  --athlete-batch 6
```

## Donde Quedan Los Resultados

El script deja:

```text
/workspace/outputs/PowerNZ_trained_models_runpod.zip
/workspace/PowerNZ_runpod_training/training_summary.json
/workspace/PowerNZ/models/powerai_bar_detector.pt
/workspace/PowerNZ/models/powerai_athlete_seg.pt
```

El `.zip` tiene los pesos y los runs completos. Los modelos se copian tambien con los nombres que espera la app:

- `models/powerai_bar_detector.pt`
- `models/powerai_athlete_seg.pt`

## Probar El Modelo En El Pod

```bash
cd /workspace/PowerNZ
python main.py \
  --input /workspace/datasets/video_prueba.mp4 \
  --output /workspace/outputs/prueba_powernz.mp4 \
  --exercise bench \
  --pose-backend yolo \
  --plate-diameter-px 120 \
  --max-frames 300 \
  --no-mobile-conversion
```

Si el detector va bien, deberia ver `Frames with reliable hub` alto. Si es bajo en banca, necesito mas ejemplos de banca con etiquetas claras de `bar_hub`.

## Descargar Resultados

Desde PowerShell:

```powershell
scp -P PUERTO root@IP_DEL_POD:/workspace/outputs/PowerNZ_trained_models_runpod.zip "C:\Users\Juanda\Downloads\"
scp -P PUERTO root@IP_DEL_POD:/workspace/outputs/prueba_powernz.mp4 "C:\Users\Juanda\Downloads\"
```

O desde JupyterLab, click derecho > Download.

## Subir Modelos A Hugging Face

Cuando tenga el `.pt` bueno en `models/`:

```bash
cd /workspace/PowerNZ
huggingface-cli login
python upload_models_to_huggingface.py --repo-id dzmbo/PowerNZ-Models
```

Si no quiero iniciar sesion en el Pod, bajo el `.zip` a mi PC, copio los `.pt` a `models/` local y ejecuto el uploader desde PowerShell.

## Apagar Para No Gastar

Cuando termine:

1. Descargo el `.zip` final.
2. Verifico que lo tengo local.
3. En RunPod hago `Stop` o `Terminate`.
4. Si use Network Volume, reviso si quiero conservarlo o borrarlo para no pagar almacenamiento.

## Mi Regla Para Tus Nuevos Datasets

- Dataset de press banca: lo usaria para mejorar `plate/bar_hub`, sobre todo los angulos donde ahora banca falla.
- Dataset de tipo de peso muerto: lo usaria despues como clasificador tecnico, no mezclado con el detector principal.

Primero entrenaria un detector multi-vista de barra con todos los ejercicios. Luego entrenaria otro modelo/clasificador para variante/forma (`sumo`, `convencional`, bloqueo dudoso, etc.).
