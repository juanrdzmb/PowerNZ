# Analisis En Kaggle

Esta es la forma mas simple de usar PowerNZ en la nube gratis.

## Antes De Empezar

Los modelos tienen que estar subidos y publicos en:

```text
https://huggingface.co/dzmbo/PowerNZ-Models
```

Si ese repo no es publico, Kaggle no podra descargar los modelos sin token.

## Pasos En Kaggle

1. Entro a [Kaggle](https://www.kaggle.com/).
2. Creo un notebook nuevo.
3. En la derecha, abro `Settings`.
4. Activo `Internet`.
5. En `Accelerator`, elijo `GPU T4` o `GPU P100` si esta disponible.
6. Subo mi video como dataset al notebook.
7. Abro o copio el notebook simple:

```text
cloud/PowerNZ_Kaggle_Simple.ipynb
```

Ese es el recomendado porque tiene una sola celda y evita errores de orden.

8. Cambio esta linea:

```python
EXERCISE = "deadlift"
```

Valores posibles:

- `deadlift`
- `squat`
- `bench`

9. Ejecuto todas las celdas.
10. Descargo el resultado desde:

```text
/kaggle/working/powernz_analizado.mp4
```

## Si No Encuentra El Video

El notebook busca automaticamente archivos `.mp4`, `.mov`, `.avi` y `.mkv` dentro de:

```text
/kaggle/input
```

Si no encuentra nada, significa que el video no quedo adjuntado como dataset del notebook.

## Si Va Lento

Reviso:

- que el acelerador sea GPU;
- que no este usando CPU;
- que el video no sea demasiado largo;
- que `--output-format portrait-720` siga activo.

Para primeras pruebas conviene usar videos cortos.
