# Como Funciona PowerAI

Esta es mi explicacion tecnica de PowerAI, escrita para poder volver al proyecto dentro de unas semanas y entender rapidamente por que cada pieza existe.

## Idea Central

PowerAI convierte un video de levantamiento en datos utiles:

- donde esta el atleta;
- donde esta el plato visible;
- donde esta el hub de la barra;
- cuanto se mueve la barra;
- a que velocidad se mueve;
- cuando empieza y termina una repeticion valida.

La app no intenta decidir como un juez humano completo. Usa una aproximacion 2D: barra + landmarks + reglas logicas inspiradas en IPF. Si el video no permite ver un angulo fiable, prefiero caer al criterio de barra antes que inventar una decision falsa.

## Pipeline

El flujo actual es de dos pasadas.

1. Normalizo cada frame al formato de salida. Por defecto uso `portrait-720`: lienzo `720x1280`, sin recortar, con el video centrado.
2. Detecto pose con YOLO Pose o MediaPipe.
3. Segmento el atleta con `powerai_athlete_seg.pt` si existe. Esa mascara no solo se dibuja: tambien se usa para corregir o bajar la confianza de landmarks que caen fuera del cuerpo.
4. Detecto barra/discos con `powerai_bar_detector.pt`, clases `plate` y `bar_hub`.
5. Filtro detecciones de barra cerca de las munecas para evitar discos del suelo o del fondo.
6. El tracker de barra estabiliza el plato visible y exige `bar_hub` fiable para medir.
7. Convierto pixeles a metros usando el diametro del disco olimpico: `0.45 m`.
8. Calculo posicion vertical, velocidad, ROM, drift horizontal y fases del levantamiento.
9. La maquina de estados cuenta reps segun ejercicio.
10. En la segunda pasada dibujo el overlay con escala de grafico estable y total de reps ya conocido.

## Modelos

Uso tres familias de modelos.

- Barra: `powerai_bar_detector.pt`, YOLO detect, clases `plate` y `bar_hub`.
- Atleta: `powerai_athlete_seg.pt`, YOLO segment, clase `athlete`.
- Pose: YOLO Pose por defecto, MediaPipe como alternativa.

En esta v1 dejo el detector entrenado como autoridad principal. La heuristica por color solo entra si la pido con `--enable-plate-heuristic` o si no hay detector entrenado.

## Mascara Y Landmarks

Antes, la mascara era sobre todo visual. Ahora la uso como una barrera de seguridad:

- si un landmark cae dentro de la mascara, lo mantengo;
- si cae justo fuera, lo acerco al pixel de mascara mas cercano;
- si cae lejos, bajo su visibilidad.

Esto hace que los calculos de rodilla, cadera y codo sean menos sensibles a personas del fondo, discos, bancos o saltos raros de pose.

## Tracking De Barra

El punto metrico es `bar_hub`, no la muneca ni el centro aproximado del plato.

El plato se puede dibujar aunque el hub no sea medible. Eso ayuda a revisar si el detector esta viendo bien el disco. Pero velocidad, trayectoria y contador solo avanzan cuando el hub pasa la compuerta de medicion.

La trayectoria se pinta fina y casi vertical. Si hay un salto horizontal grande, el renderer corta el segmento para que no aparezca una linea falsa cruzando el video.

## Conteo De Repeticiones

Uso dos maquinas de estado:

- `deadlift`: movimiento concentrico primero, de abajo hacia arriba.
- `squat` y `bench`: movimiento excentrico primero, baja y luego sube.

Para aceptar un bloqueo no basta con que la velocidad sea casi cero. Tambien exijo:

- rango suficiente;
- fase concentrica suficientemente madura;
- `lockout_ok` si la pose puede decidir.

Esto evita contar pausas tempranas como reps.

## Criterios IPF Aproximados

Los criterios que puedo estimar con video 2D son:

- Sentadilla: rodilla/cadera para profundidad y bloqueo.
- Peso muerto: rodilla/cadera extendidas al final.
- Press banca: recorrido minimo de barra y codos bloqueados arriba.

No intento resolver señales, contacto con pecho, pies, gluteos, desplazamientos laterales o apoyo indebido como lo haria un arbitro real. Lo documento asi para no vender precision que el video 2D no puede garantizar.

## Overlay

La pantalla final queda deliberadamente simple:

- silueta clara del atleta;
- esqueleto biomecanico;
- caja `Plate`;
- caja `Bar` si hay hub fiable;
- trayectoria fina de barra;
- panel de velocidad/ROM/reps/drift;
- grafico inferior de velocidad de barra;
- velocidades corporales como indicadores compactos, no como lineas locas.

El modo `--velocity-chart multi` existe para depurar, pero no es el default visual de v1.

## Salida De Video

Por defecto exporto `720x1280`. Si el video original es horizontal o raro, lo encajo dentro del lienzo sin recortar. Esto mantiene al atleta, la barra y los discos completos.

Para depurar puedo usar:

```powershell
--output-format source
```

## Limitaciones

- La validez IPF es una ayuda tecnica, no un veredicto oficial.
- Un mal angulo de camara puede ocultar articulaciones.
- Si el hub no se detecta, no hay velocidad fiable.
- Si el modelo de barra no generaliza a un gimnasio o disco nuevo, toca reentrenar con mas datos.

## Archivos Clave

- `main.py`: orquesta el analisis de dos pasadas.
- `io_video.py`: lectura, escritura y formato 720x1280.
- `detect_objects.py`: detector de barra y fallback de color.
- `bar_anchor.py`: tracking del plato/hub.
- `segmentation.py`: mascara del atleta.
- `pose.py`: pose y refinamiento con mascara.
- `metrics.py`: velocidad y maquinas de estado.
- `biomech_angles.py`: angulos para compuertas IPF.
- `render_overlay.py`: HUD final.
