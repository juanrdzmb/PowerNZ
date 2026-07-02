# Como Funciona PowerNZ

Esta es mi explicacion tecnica de PowerNZ, escrita para poder volver al proyecto dentro de unas semanas y entender rapidamente por que cada pieza existe.

## Idea Central

PowerNZ convierte un video de levantamiento en datos utiles:

- donde esta el atleta;
- donde esta el plato visible;
- donde esta el hub de la barra;
- cuanto se mueve la barra;
- a que velocidad se mueve;
- cuando empieza y termina una repeticion valida.

La app no intenta decidir como un juez humano completo. Usa una aproximacion 2D: barra + landmarks + reglas logicas inspiradas en IPF. Si el video no permite ver un angulo fiable, prefiero no marcar la repeticion como valida antes que inventar una decision falsa.

## Pipeline

El flujo actual es de dos pasadas.

1. Normalizo cada frame al formato de salida. Por defecto uso `portrait-720`: lienzo `720x1280`, sin recortar, con el video centrado.
2. En modo automático, YOLO mantiene la identidad del atleta y MediaPipe aporta pose y máscara cuando sus landmarks coinciden con esa persona.
3. Segmento el atleta con `powerai_athlete_seg.pt` si existe y selecciono la máscara más coherente con los landmarks. Esa máscara no solo se dibuja: también corrige o baja la confianza de puntos fuera del cuerpo.
4. Detecto barra/discos con `powerai_bar_detector.pt`, clases `plate` y `bar_hub`.
5. Filtro detecciones de barra cerca de las munecas para evitar discos del suelo o del fondo.
6. El tracker de barra estabiliza el plato visible y mide desde `bar_hub`; en lateral admite el centro de un disco fiable como eje de respaldo.
7. Convierto pixeles a metros usando el diametro del disco olimpico: `0.45 m`.
8. Reconstruyo la trayectoria del eje de barra al terminar la primera pasada: filtro saltos, mantengo huecos reales y calculo velocidad centrada sin desfase.
9. Reproduzco las reglas del ejercicio y marco `review` si la evidencia de cámara/pose no es suficiente.
10. En la segunda pasada dibujo el overlay con escala de gráfico estable, total conocido y máscaras cacheadas de la primera pasada.

## Modelos

Uso tres familias de modelos.

- Barra: `powerai_bar_detector.pt`, YOLO detect, clases `plate` y `bar_hub`.
- Atleta: `powerai_athlete_seg.pt`, YOLO segment, clase `athlete`.
- Pose: fusión automática de identidad YOLO y MediaPipe; cada backend sigue disponible para depuración.

En esta v1 dejo el detector entrenado como autoridad principal. La heuristica por color solo entra si la pido con `--enable-plate-heuristic` o si no hay detector entrenado.

## Mascara Y Landmarks

Antes, la mascara era sobre todo visual. Ahora la uso como una barrera de seguridad:

- si un landmark cae dentro de la mascara, lo mantengo;
- si cae justo fuera, lo acerco al pixel de mascara mas cercano;
- si cae lejos, bajo su visibilidad.

Esto hace que los calculos de rodilla, cadera y codo sean menos sensibles a personas del fondo, discos, bancos o saltos raros de pose.

## Tracking De Barra

El punto metrico principal es `bar_hub`. En vista lateral, si el hub queda oculto pero el detector mantiene un disco fiable, el centro de ese disco se usa como eje de la barra; disco y barra comparten altura y así la trayectoria no salta a las muñecas del atleta. En vistas diagonal o frontal se conserva el respaldo corporal cuando no hay otro punto medible.

El plato se puede dibujar aunque no sea medible. En lateral, un disco detectado con confianza suficiente puede pasar la compuerta como `plate_center`; en los demás ángulos se exige el hub o se marca explícitamente el respaldo corporal. Esto permite revisar qué fuente produjo la trayectoria sin confundir una caja visual con una medición fiable.

La trayectoria se pinta fina y casi vertical. Si se pierden tanto el hub como el disco lateral fiable, hay un salto horizontal grande o cambia el lado detectado, guardo un corte explícito y el renderer no une esos puntos.

## Conteo De Repeticiones

Uso dos maquinas de estado:

- `deadlift`: movimiento concentrico primero, de abajo hacia arriba.
- `squat` y `bench`: movimiento excentrico primero, baja y luego sube.

Para aceptar un bloqueo no basta con que la velocidad sea casi cero. Tambien exijo:

- rango suficiente;
- fase concentrica suficientemente madura;
- `lockout_ok` con evidencia de pose cuando la regla lo necesita;
- que no haya bajada de la barra antes del bloqueo.

Esto evita contar pausas tempranas como reps.

## Criterios IPF Aproximados

Los criterios que puedo estimar con video 2D son:

- Sentadilla: rodilla/cadera para profundidad y bloqueo.
- Peso muerto: rodilla/cadera extendidas al final y sin bajada antes de terminar.
- Press banca: codo al nivel de hombro en la bajada, recorrido minimo de barra y codos bloqueados arriba.

No intento resolver señales, contacto con pecho, pies, gluteos, desplazamientos laterales o apoyo indebido como lo haria un arbitro real. Lo documento asi para no vender precision que el video 2D no puede garantizar.

Por defecto uso validacion estricta: si un landmark clave no se ve, esa parte de la regla queda como no confiable y la rep no se acepta automaticamente. Para depurar clips dificiles puedo usar `--no-strict-ipf-validation`, pero no es el flujo normal de v1.

## Overlay

La pantalla final queda deliberadamente simple:

- silueta clara del atleta;
- esqueleto biomecanico;
- caja `Plate`;
- caja `Bar` si hay hub fiable o un disco lateral usado como eje;
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
- Si no se detecta ni hub ni un disco lateral fiable, no hay velocidad de barra fiable.
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

## Carga Manual

Quite la estimacion normal por color porque no era fiable. La app no sabe cuantos kilos hay en la barra solo mirando discos si no tiene un modelo entrenado para pesos por disco. Ahora `CARGA` se oculta por defecto y solo aparece si yo paso un valor manual:

```powershell
--load-kg 180
```

Para estimar peso automaticamente en el futuro necesitare otro dataset/modelo de discos por peso o una entrada guiada por el usuario.
