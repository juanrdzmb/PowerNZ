# Bitacora De PowerAI

Escribo esta bitacora para no perder el hilo de los problemas que fueron apareciendo y como los fui resolviendo.

## Mascara Del Atleta

Al principio la mascara era mas visual que tecnica. Se veia bien en algunos frames, pero no ayudaba lo suficiente a que los landmarks se quedaran sobre el cuerpo. Con el modelo `powerai_athlete_seg.pt` cambie el flujo: ahora la mascara entrenada es la principal y tambien sirve para corregir o bajar la confianza de landmarks que caen fuera del atleta.

## Caja Del Plato

Habia momentos en que el rectangulo del plato no abrazaba bien el disco porque se mezclaban cajas del modelo, rectangulos refinados y heuristica por color. La v1 deja el detector entrenado como autoridad principal: `plate` dibuja el disco y `bar_hub` mide la barra. La heuristica de color queda fuera por defecto cuando el modelo carga.

## Velocidad Y Trayectoria

El grafico se estaba volviendo demasiado ruidoso con varias lineas de landmarks. Lo simplifique: la grafica inferior muestra la velocidad de la barra como metrica principal y las velocidades corporales quedan como indicadores compactos. La trayectoria tambien queda fina y se corta si hay saltos raros.

## Conteo De Repeticiones

El contador podia aceptar pausas tempranas como bloqueo. Lo corregi combinando rango suficiente, fase concentrica madura y bloqueo tecnico por pose cuando la pose puede decidir. Para sentadilla y banca mantengo una maquina diferente porque esos ejercicios bajan antes de subir.

## Formato Del Video

Necesitaba una salida consistente para revisar rapido y preparar contenido. Ahora uso `720x1280` por defecto, sin recortar. Si el video no es vertical, lo encajo completo en el lienzo.

## Git Y Version

El repo quedo en medio de un rebase con conflictos. La limpieza de v1 empieza resolviendo ese estado y separando el trabajo nuevo en una rama limpia. No voy a etiquetar `v1.0.0` hasta revisar capturas y videos finales.
