# Modelos Y Descarga

Los modelos entrenados no viven dentro de Git. Mi idea es que el repo siga ligero y que los pesos se descarguen cuando hagan falta.

## Opcion Recomendada: Hugging Face

Para usuarios normales, lo mas limpio es publicar los pesos en un repo publico de Hugging Face:

```text
dzmbo/PowerNZ-Models
```

Archivos que debe contener:

- `powerai_bar_detector.pt`
- `powerai_athlete_seg.pt`
- `pose_landmarker_lite.task` opcional

Hugging Face funciona muy bien aqui porque el usuario puede hacer `git clone` de PowerNZ y luego descargar modelos con una URL publica, sin iniciar sesion en GitHub ni tener permisos especiales.

## Primer Setup En Hugging Face

1. Creo o entro a mi cuenta en [Hugging Face](https://huggingface.co/).
2. Creo un token con permiso de escritura en [Access Tokens](https://huggingface.co/settings/tokens).
3. Instalo dependencias si todavia no lo hice:

```powershell
python -m pip install -r requirements.txt
```

4. Inicio sesion desde PowerShell:

```powershell
hf auth login
```

Si mi instalacion usa el comando antiguo, tambien sirve:

```powershell
huggingface-cli login
```

5. Subo los modelos:

```powershell
python upload_models_to_huggingface.py
```

Ese script crea el repo `dzmbo/PowerNZ-Models` si no existe y sube los tres archivos desde:

```text
C:\Users\Juanda\Documents\PowerAI\models
```

Si quiero usar otro usuario u organizacion:

```powershell
python upload_models_to_huggingface.py --repo-id TU_USUARIO/PowerNZ-Models
```

Despues actualizo las URLs de `models/model_manifest.json` si cambia el `repo-id`.

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

Tambien puedo hacer doble clic en:

```text
descargar_modelos.bat
```

## Flujo Para Un Usuario Nuevo

```powershell
git clone https://github.com/juanrdzmb/PowerNZ.git
cd PowerNZ
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python model_downloader.py
python powernz_launcher.py
```

Si usa Windows, tambien puede abrir `abrir_powernz.bat`; si faltan modelos, la interfaz pregunta si los descarga.

## Respaldo: GitHub Releases

Tambien tengo una release privada `models-v1` en GitHub. Sirve para mi trabajo interno, pero no es ideal para usuarios finales porque una release privada necesita autenticacion. Si quiero usar GitHub Releases sin friccion, tendria que hacer publica la release o el repo.

## Videos De Test

Los videos de prueba tampoco deberian ir en Git. Para compartirlos puedo usar:

- Hugging Face Dataset si quiero dejarlos publicos y ordenados;
- Google Drive/Dropbox si son privados;
- GitHub Release si son pocos y no pesan demasiado.
