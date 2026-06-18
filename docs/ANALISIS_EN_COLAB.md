# Analisis En Google Colab

Uso Colab si Kaggle no me deja ejecutar el notebook.

## Pasos

1. Abro Google Colab.
2. Subo este notebook:

```text
cloud/PowerNZ_Colab_Analisis.ipynb
```

3. En Colab voy a:

```text
Entorno de ejecucion > Cambiar tipo de entorno de ejecucion
```

4. Elijo `GPU` si esta disponible.
5. Ejecuto las celdas en orden.
6. Cuando aparezca el selector de archivo, subo mi video.
7. Al final descarga `powernz_analizado.mp4`.

## Que Tengo Que Cambiar

En la primera celda puedo cambiar:

```python
EXERCISE = "deadlift"
```

Opciones:

- `deadlift`
- `squat`
- `bench`

## Importante

Los modelos deben estar publicos en:

```text
https://huggingface.co/dzmbo/PowerNZ-Models
```

Si no estan publicos, Colab no podra descargarlos sin token.
