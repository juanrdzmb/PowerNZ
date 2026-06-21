# PowerNZ web beta

La web es una beta privada para muy pocas personas. No crea cuentas de usuario: cada análisis queda protegido por una cookie privada del navegador que lo subió.

## Probarla en este PC

La preparación solo hace falta una vez:

```powershell
python -m pip install -r requirements.txt
python model_downloader.py
```

Después, para abrirla:

```powershell
python -m web
```

Abre `http://127.0.0.1:8000` en el navegador. Para detenerla, vuelve a esa ventana y pulsa `Ctrl+C`.

La carpeta `web_data/` contiene las subidas, resultados y la cola local. No se versiona. El servicio borra cada análisis automáticamente después de 24 horas; el botón **Eliminar ahora** lo borra antes.

## Límites deliberados de la beta

- MP4 o MOV; hasta 250 MB, 60 segundos y 1080p.
- Solo un análisis se ejecuta a la vez.
- Máximo dos subidas por hora desde la misma conexión.
- No se guardan vídeos para entrenar modelos ni para investigar feedback. Los comentarios y metadatos mínimos de feedback vencen en 30 días.

## Publicarla en Google Cloud

La imagen Docker incluye FFmpeg y los modelos de PowerNZ. En la VM europea se usará Docker Compose junto a Caddy, que proporciona HTTPS.

No publicar todavía desde este documento: primero hay que revisar la web local y configurar en Google Cloud la VM, firewall, presupuesto y dirección IP. Esa parte se hará guiada paso a paso para no crear recursos adicionales ni cargos inesperados.
