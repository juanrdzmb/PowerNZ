# Modelos Y Descarga

Los modelos entrenados no viven dentro de Git. Mi idea es mantener el codigo ligero y publicar los pesos como archivos descargables.

## Donde Los Subo

Para esta v1 uso GitHub Releases:

- repo: `juanrdzmb/PowerNZ`
- tag de modelos: `models-v1`
- assets esperados:
  - `powerai_bar_detector.pt`
  - `powerai_athlete_seg.pt`
  - `pose_landmarker_lite.task` opcional

GitHub Releases me sirve bien para estos pesos porque cada archivo pesa menos de 2 GB. Si en el futuro los modelos crecen mucho, o quiero tener una pagina mas clara para modelos, Hugging Face Hub seria la alternativa natural.

## Como Los Baja La App

El archivo `models/model_manifest.json` guarda:

- nombre del modelo;
- ruta local esperada;
- URL de descarga;
- hash SHA256;
- si es obligatorio u opcional.

Cuando ejecuto:

```powershell
python model_downloader.py
```

la app descarga lo que falta, verifica el hash y deja cada archivo en `models/`.

Si la release es privada, la descarga directa puede fallar. En ese caso el descargador intenta usar GitHub CLI, asi que necesito haber iniciado sesion con:

```powershell
gh auth login
```

Para usuarios finales lo mas sencillo es que los modelos esten en una release publica o en un repositorio publico de Hugging Face.

## Como Cambiar De Hosting

Si muevo los pesos a Hugging Face, Drive u otro sitio, no tengo que cambiar codigo. Solo actualizo las URLs en:

```text
models/model_manifest.json
```

Mantengo los mismos nombres de archivo para que el resto de PowerNZ no cambie.
