# APLICACIÓN DE MODELOS DE MACHINE LEARNING PARA LA DETECCIÓN Y CLASIFICACIÓN DEL DETERIORO COGNITIVO MEDIANTE EL USO EVALUACIONES NEUROPSICOLÓGICAS EN ADULTOS MAYORES EN COLOMBIA

La vejez trae consigo una serie de cambios y reestructuración estructurales que incrementa la prevalencia de enfermedades neurocognitivas como el deterioro cognitivo leve (DCL) (Gauthier et al., 2006), en Colombia el acceso limitado a especialistas y las características socioculturales del país representan una dificultad en el rápido diagnóstico de esta condición. Ante este escenario, este trabajo pretende responder la pregunta ¿Cómo se desempeñan las herramientas de machine learning en la identificación temprana del deterioro cognitivo en adultos mayores en contextos colombianos? 

 El propósito general del proyecto es realizar una comparación entre modelos construidos y entrenados para la detección y clasificación del deterioro cognitivo usando datos de pruebas neuropsicológicas especializadas para cada dominio cognitivo tomadas en pacientes en Colombia. Para ello, se plantean una serie de objetivos que van desde el estudio de estas pruebas neuropsicológicas, la construcción de un pipeline de datos, la aplicación de técnicas de inteligencia artificial y aprendizaje supervisado y evaluar su desempeño mediante métricas de precisión y robustez. Este proyecto se define en el marco de una investigación aplicada de diseño observacional y no experimental, que está basada en el análisis de datos secundarios provenientes de instituciones médicas de Caldas. El proyecto sigue los lineamientos de la metodología CRISP-DM, la cual define toda la estructura de un proyecto fundamentado en la ciencia de datos.  

Los resultados del proyecto incluyen tanto la fundamentación conceptual de las evaluaciones neuropsicológicas, un pipeline entre los datos en bruto de los test neuropsicológicos a un conjunto de datos con requerimientos necesarios para la incorporación de modelos de ML y la identificación de patrones complejos de deterioro cognitivo y la selección de un modelo que presente un alto rendimiento predictivo. Una de las conclusiones principales es resaltar cómo el aprendizaje automático puede constituir una estrategia ética y viable para apoyar la toma de decisiones médicas basadas en datos en un contexto colombiano.  


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

Los notebooks en `src/notebooks/` se mantienen como referencia exploratoria del proceso, pruebas visuales y avances exploratorios. Sin embargo, se prevee alimentar el proyecto de acuerdo a los requerimentos y objetivos del mismo.

## Notas

- El pipeline conserva pacientes y evita eliminar filas por nulos.
- La imputación se hace por grupo clínico `dc` cuando la tasa de nulos lo permite.
- `df_complete` está pensado para alimentar gráficas, estadística inferencial y modelos de clasificación.