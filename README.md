# APLICACIÓN DE MODELOS DE MACHINE LEARNING PARA LA DETECCIÓN Y CLASIFICACIÓN DEL DETERIORO COGNITIVO MEDIANTE EL USO EVALUACIONES NEUROPSICOLÓGICAS EN ADULTOS MAYORES EN COLOMBIA 

Pipeline de extracción, limpieza e integración de datos neuropsicológicos para análisis exploratorio, visualización y modelado.

El proyecto parte de un archivo Excel alojado en GitHub, procesa dos formatos de tabla distintos y construye un dataset final listo para gráficas y modelos.

## Contenido del proyecto

- `src/etl.py`: módulo principal del ETL.
- `src/notebooks/`: notebooks de exploración y desarrollo del flujo.
- `pyproject.toml`: dependencias del proyecto y configuración básica.

## Qué hace el ETL

El pipeline implementado en `src/etl.py`:

- descarga el Excel desde GitHub usando credenciales en `.env`;
- limpia las hojas vacías;
- detecta si cada hoja corresponde a formato de Tabla 0 o Tabla 1;
- extrae los indicadores clínicos de cada paciente;
- normaliza valores categóricos;
- imputa valores nulos con criterio clínico por grupo `dc`;
- construye `df_complete` con los dominios cognitivos agregados.

La función principal es `run_etl()` y devuelve:

1. `df_tabla_0_imp`
2. `df_tabla_1_imp`
3. `df_complete`

## Requisitos

- Python `>= 3.14`
- Conexión al repositorio o archivo Excel configurado en GitHub

Dependencias principales:

- `pandas`
- `numpy`
- `requests`
- `openpyxl`
- `python-dotenv`
- `matplotlib`
- `seaborn`
- `scipy`

## Instalación

```bash
uv sync
```

Si no usas `uv`, instala las dependencias con tu gestor habitual a partir de `pyproject.toml`.

## Uso

Ejecutar el ETL desde Python:

```python
from src.etl import run_etl

df_tabla_0_imp, df_tabla_1_imp, df_complete = run_etl(verbose=True)
```

También puedes usar la función de carga directa si quieres cambiar la ruta del `.env`:

```python
df_tabla_0_imp, df_tabla_1_imp, df_complete = run_etl(dotenv_path=".env", verbose=True)
```

## Salidas del pipeline

- `df_tabla_0_imp`: tabla 0 normalizada e imputada.
- `df_tabla_1_imp`: tabla 1 normalizada e imputada.
- `df_complete`: dataset integrado con dominios cognitivos y la columna `age_num` lista para análisis.

## Notebooks

Los notebooks en `src/notebooks/` se mantienen como referencia exploratoria del proceso y para pruebas visuales. El flujo reutilizable debe vivir en `src/etl.py`.

## Notas

- El pipeline conserva pacientes y evita eliminar filas por nulos.
- La imputación se hace por grupo clínico `dc` cuando la tasa de nulos lo permite.
- `df_complete` está pensado para alimentar gráficas, estadística inferencial y modelos de clasificación.