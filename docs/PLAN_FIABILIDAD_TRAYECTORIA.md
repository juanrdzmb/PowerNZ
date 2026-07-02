# Plan de fiabilidad de trayectoria y velocidad

## Principio

Entrenar otro modelo no corrige una máquina de estados, una derivada ruidosa ni una interfaz que muestra datos futuros. PowerNZ mantiene separados tres problemas:

1. **Percepción**: localizar atleta, disco y centro/eje de barra.
2. **Seguimiento y biomecánica**: reconstruir posición, velocidad, fases y repeticiones.
3. **Presentación**: mostrar únicamente información disponible en ese instante.

Los modelos actuales siguen siendo la base. Solo se reentrenan cuando una revisión de vídeos demuestra fallos de percepción después de estabilizar el código.

## Cambios aplicados

- El centro de barra es la medida principal; en lateral, el centro del disco aporta una corrección vertical limitada.
- Los saltos aislados del detector se eliminan antes de calcular la derivada.
- Las coordenadas de trayectoria se reconstruyen y suavizan sin inventar datos durante oclusiones.
- La trayectoria comienza al iniciar el movimiento, se conserva al terminar y se reinicia en la repetición siguiente.
- Una oscilación pequeña en la parte alta se considera bloqueo, no una bajada real.
- En sentadilla y banca, el tiempo excéntrico comienza cuando empieza el descenso, no al terminar la subida.
- La tabla `FASTEST` solo muestra repeticiones ya bloqueadas en el fotograma actual. La mejor repetición se recalcula después de cada bloqueo.
- El gráfico utiliza exactamente la velocidad estabilizada que consume la máquina de estados.
- La vista lateral incorpora un eje hombro-cadera para revisar el ángulo del torso.
- La silueta pasa a grafito translúcido con contorno fino; se eliminan el halo cian y las etiquetas corporales por defecto.

## Cuándo mejorar los modelos

Reentrenar el detector únicamente si persisten alguno de estos casos:

- el disco cargado no aparece durante varios fotogramas consecutivos;
- se elige un disco del suelo o del rack;
- el `bar_hub` salta a otra barra o desaparece con ángulos frecuentes;
- la caja cambia mucho de tamaño sin movimiento real.

Reentrenar el segmentador únicamente si:

- la máscara cambia del atleta a un spotter;
- pierde grandes partes del cuerpo de forma repetida;
- incluye banco, discos o fondo como parte del atleta.

Los nuevos fotogramas deben salir de esos fallos reales. Conviene añadir negativos difíciles y mantener cada vídeo completo dentro de un único split para no falsear la validación.

## Criterios de aceptación por vídeo

- ninguna repetición aparece en `FASTEST` antes del bloqueo;
- el mejor `R#` cambia solo cuando termina una repetición más rápida;
- no hay velocidad negativa en bloqueo por una oscilación inferior a 1,5 cm;
- una oclusión corta crea un hueco o una continuación conservadora, nunca un salto;
- la trayectoria visible pertenece a una sola repetición;
- velocidad media, pico, ROM y tiempos proceden de la misma señal estabilizada;
- lateral, diagonal y frontal se revisan por separado.

## Siguiente validación con material real

Ejecutar al menos tres vídeos por ejercicio y ángulo. Para cada fallo, guardar el fotograma, la fuente usada (`bar_hub`, `plate_center` o `body_proxy`) y el informe JSON/CSV. Si el fallo está en la caja, se corrige con datos/modelo; si la caja es correcta pero la métrica no, se corrige en código.
