# APLICACIÓN DE MODELOS DE MACHINE LEARNING PARA LA DETECCIÓN Y CLASIFICACIÓN DEL DETERIORO COGNITIVO MEDIANTE EL USO EVALUACIONES NEUROPSICOLÓGICAS EN ADULTOS MAYORES EN COLOMBIA

La vejez trae consigo una serie de cambios y reestructuración estructurales que incrementa la prevalencia de enfermedades neurocognitivas como el deterioro cognitivo leve (DCL) (Gauthier et al., 2006), en Colombia el acceso limitado a especialistas y las características socioculturales del país representan una dificultad en el rápido diagnóstico de esta condición. Ante este escenario, este trabajo pretende responder la pregunta ¿Cómo se desempeñan las herramientas de machine learning en la identificación temprana del deterioro cognitivo en adultos mayores en contextos colombianos?

El propósito general del proyecto es realizar una comparación entre modelos construidos y entrenados para la detección y clasificación del deterioro cognitivo usando datos de pruebas neuropsicológicas especializadas para cada dominio cognitivo tomadas en pacientes en Colombia. Para ello, se plantean una serie de objetivos que van desde el estudio de estas pruebas neuropsicológicas, la construcción de un pipeline de datos, la aplicación de técnicas de inteligencia artificial y aprendizaje supervisado y evaluar su desempeño mediante métricas de precisión y robustez. Este proyecto se define en el marco de una investigación aplicada de diseño observacional y no experimental, que está basada en el análisis de datos secundarios provenientes de instituciones médicas de Caldas. El proyecto sigue los lineamientos de la metodología CRISP-DM, la cual define toda la estructura de un proyecto fundamentado en la ciencia de datos.

Los resultados del proyecto incluyen tanto la fundamentación conceptual de las evaluaciones neuropsicológicas, un pipeline entre los datos en bruto de los test neuropsicológicos a un conjunto de datos con requerimientos necesarios para la incorporación de modelos de ML y la identificación de patrones complejos de deterioro cognitivo y la selección de un modelo que presente un alto rendimiento predictivo. Una de las conclusiones principales es resaltar cómo el aprendizaje automático puede constituir una estrategia ética y viable para apoyar la toma de decisiones médicas basadas en datos en un contexto colombiano.

## Contenido del proyecto

- `src/etl.py`: script de ETL original/de respaldo.
- `src/services/`: paquete modular de servicios de ETL:
  - `__init__.py`: API pública del paquete.
  - `config.py`: constantes de dominio, features y mapeos.
  - `utils.py`: funciones utilitarias de normalización de texto.
  - `cleaning.py`: limpieza de hojas y detección de formato.
  - `features.py`: extracción de información de pacientes y features.
  - `normalization.py`: normalización de variables categóricas.
  - `imputation.py`: imputación clínica de valores nulos.
  - `etl.py`: orquestador principal que ejecuta y coordina todos los pasos del ETL.
- `src/notebooks/`: notebooks de exploración y desarrollo del flujo.
- `pyproject.toml`: dependencias del proyecto y configuración básica.

## Qué hace el ETL

El pipeline implementado en `src/services/etl.py` (y expuesto a través de `src/services`):

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

- **Python `>= 3.14`**
- **uv** (Recomendado para la gestión de dependencias y entornos virtuales rápidos) o **Python venv + pip**
- Acceso a la fuente de datos (requiere un token de GitHub si el repositorio del que descarga los datos es privado o para evitar rate-limiting)

### Dependencias principales

Las librerías del proyecto definidas en [pyproject.toml](file:///e:/Projects/dc_model_ts/pyproject.toml) son:

- `pandas`
- `numpy`
- `requests`
- `openpyxl`
- `python-dotenv`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `shap`

---

## Instalación y Configuración del Entorno

Sigue uno de los dos métodos a continuación para preparar tu entorno de ejecución:

### Opción 1: Usando `uv` (Recomendado)

[uv](https://github.com/astral-sh/uv) es un instalador y gestor de paquetes de Python extremadamente rápido.

1. **Instalar `uv`** (si aún no lo tienes):
   - **En Windows (PowerShell):**
     ```powershell
     irm https://astral.sh/uv/install.ps1 | iex
     ```
   - **En macOS/Linux:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. **Crear y sincronizar el entorno virtual**:
   Ejecuta el siguiente comando en la raíz del proyecto. Este creará la carpeta `.venv`, descargará la versión correcta de Python configurada en `.python-version` y sincronizará todas las dependencias:

   ```bash
   uv sync
   ```

3. **Activar el entorno virtual**:
   - **En Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **En macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

---

### Opción 2: Usando `venv` y `pip` tradicionales

Si prefieres no usar `uv`, puedes configurar el entorno usando las herramientas integradas de Python:

1. **Crear el entorno virtual**:

   ```bash
   python -m venv .venv
   ```

2. **Activar el entorno virtual**:
   - **En Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **En Windows (CMD):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **En macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Actualizar pip e instalar dependencias**:
   ```bash
   python -m pip install --upgrade pip
   # Instala el proyecto en modo editable con todas las dependencias
   pip install -e .
   ```

---

### Configuración de Variables de Entorno (`.env`)

El pipeline de descarga requiere credenciales para conectarse a GitHub y obtener el dataset.

1. Copia el archivo de plantilla `.env.example` y renombralo a `.env`:

   ```bash
   cp .env.example .env
   ```

   _(En Windows PowerShell puedes usar: `Copy-Item .env.example .env`)_

2. Abre el archivo `.env` recién creado y configura tus credenciales:
   - `GITHUB_TOKEN`: Tu token de acceso personal (PAT) de GitHub.
   - `DATA_FILE_PATH`: La URL directa (raw) del archivo Excel con los datos.

## Uso del ETL

Ejecutar el ETL modularizado desde Python:

```python
from src.services import run_etl

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
