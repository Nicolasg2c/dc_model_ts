# APLICACIÓN DE MODELOS DE MACHINE LEARNING PARA LA DETECCIÓN Y CLASIFICACIÓN DEL DETERIORO COGNITIVO MEDIANTE EL USO EVALUACIONES NEUROPSICOLÓGICAS EN ADULTOS MAYORES EN COLOMBIA

La vejez trae consigo una serie de cambios y reestructuración estructurales que incrementa la prevalencia de enfermedades neurocognitivas como el deterioro cognitivo (DC) (Gauthier et al., 2006), en Colombia el acceso limitado a especialistas y las características socioculturales del país representan una dificultad en el rápido diagnóstico de esta condición. Ante este escenario, este trabajo pretende responder la pregunta ¿Cómo se desempeñan las herramientas de machine learning en la identificación del deterioro cognitivo en adultos mayores en contextos colombianos?

El propósito general del proyecto es realizar una comparativa entre modelos construidos y entrenados para la detección y clasificación del deterioro cognitivo usando datos de pruebas neuropsicológicas especializadas en cada uno de los dominios cognitivos, tomadas en personas de edad avanzada en Colombia. Para ello, se plantean una serie de objetivos que van desde el estudio de estas pruebas neuropsicológicas, la construcción de un pipeline de datos, la aplicación de técnicas de aprendizaje supervisado y evaluar su desempeño mediante métricas de precisión y robustez. Este proyecto se define en el marco de una investigación aplicada de diseño observacional y no experimental, que está basada en el análisis de datos secundarios provenientes de instituciones médicas de Caldas. El proyecto sigue los lineamientos de la metodología CRISP-DM, la cual define toda la estructura de un proyecto fundamentado en la ciencia de datos.

Los resultados del proyecto incluyen tanto la fundamentación conceptual de las evaluaciones neuropsicológicas, un pipeline desde los datos en bruto de las pruebas neuropsicológicos a un conjunto de datos con requerimientos necesarios para la incorporación de modelos de machine learning y la identificación de patrones complejos de deterioro cognitivo. Una de las conclusiones principales es resaltar cómo el aprendizaje automático puede constituir una estrategia ética y viable para apoyar la toma de decisiones médicas basadas en datos en un contexto colombiano.

## Contenido del proyecto

- `src/pipeline/`: paquete modular de servicios de ETL:
  - `__init__.py`: API pública del paquete.
  - `config.py`: constantes de dominio, features y mapeos.
  - `utils.py`: funciones utilitarias de normalización de texto.
  - `cleaning.py`: limpieza de hojas y detección de formato.
  - `features.py`: extracción de información de pacientes y features.
  - `normalization.py`: normalización de variables categóricas.
  - `imputation.py`: imputación clínica de valores nulos.
  - `etl.py`: orquestador principal que ejecuta y coordina todos los pasos del ETL.
- `src/modeling/`: paquete modular de entrenamiento, evaluación e interpretabilidad de modelos:
  - `__init__.py`: API pública del paquete de modeling.
  - `config.py`: semilla de replicabilidad, métricas de scoring personalizadas y esquemas de validación cruzada.
  - `utils.py`: funciones de entrenamiento, evaluación, visualización e interpretabilidad de modelos.
- `models/`: modelos de Machine Learning entrenados y serializados con `joblib`.
- `notebooks/`: notebooks de exploración, modelado e interpretabilidad.
- `pyproject.toml`: dependencias del proyecto y configuración básica.

## Qué hace el ETL

El pipeline implementado en `src/pipeline/etl.py` (y expuesto a través de `src/pipeline`):

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
from src.pipeline import run_etl

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

---

## Módulo de Modeling (`src/modeling`)

El paquete `src/modeling` centraliza toda la lógica de entrenamiento, evaluación e interpretabilidad de los modelos de Machine Learning. Se importa directamente desde los notebooks de modelado.

### `src/modeling/config.py` — Configuración global del modelado

| Objeto              | Tipo                      | Descripción                                                                                                                                                       |
| ------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seed`              | `int`                     | Semilla global (`19971711`) para garantizar reproducibilidad en todos los experimentos.                                                                           |
| `scoring`           | `dict`                    | Diccionario de métricas personalizadas para `cross_validate`: F1-Macro, Balanced Accuracy, Sensibilidad y Especificidad por clase (Control=0, DCL=1, Demencia=2). |
| `cv_repeated_kfold` | `RepeatedStratifiedKFold` | Validación cruzada estratificada repetida (5 folds × 20 repeticiones). Usada en comparaciones robustas y pruebas de Wilcoxon.                                     |
| `cv_kfold`          | `StratifiedKFold`         | Validación cruzada estratificada simple (5 folds). Usada para matrices de confusión y curvas ROC.                                                                 |

### `src/modeling/utils.py` — Funciones de modelado

A continuación se describen todas las funciones expuestas por el módulo:

#### `train_model(model, X, y, cv)`

Construye un `Pipeline` de scikit-learn (imputación por mediana → escalado estándar → clasificador) y ejecuta `cross_validate` con el `scoring` personalizado definido en `config.py`. Devuelve el diccionario de scores por fold.

#### `get_confusion_matrix(models, X, y, cv_kfold)`

Genera y visualiza la matriz de confusión para cada modelo del diccionario `{nombre: modelo}` usando predicciones Out-of-Fold (`cross_val_predict`). Etiquetas de clase: `["Control", "DCL", "Demencia"]`.

#### `print_scores(models, X, y, cv_kfold)`

Entrena cada modelo con `train_model` y muestra dos tablas:

- **Resultados principales**: F1-Macro, Balanced Accuracy, Sensibilidad y Especificidad por clase (media ± std).
- **Análisis de overfitting**: F1 en entrenamiento vs. validación y la brecha (gap) con su desviación estándar.

#### `compare_datasets_wilcoxon(models, X_mean, y_mean, X_median, y_median, cv, metrics, plot_results)`

Comparador estadístico entre dos conjuntos de datos (construidos con la **media** y la **mediana** como estrategia de agregación de dominios). Para cada modelo y métrica aplica la **prueba de Wilcoxon de rangos con signo** (bilateral, α=0.05). Devuelve un `pd.DataFrame` con el estadístico W, p-valor, dirección de la diferencia y conclusión, y genera gráficos de caja comparativos.

#### `graficar_curvas_roc_multiclase(model, model_name, X, y, cv, class_names)`

Calcula y grafica las curvas ROC para cada clase usando la estrategia **One-vs-Rest (OvR)** con predicciones Out-of-Fold. Reporta el AUC por clase y el Macro-AUC general. Devuelve un diccionario `{clase: AUC}`.

#### `nested_cross_validation_multi(pipeline, parametros, X, y, cv_interno, cv_externo, scoring, metrica_optimizacion)`

Implementa **Nested Cross-Validation** completa:

- **Ciclo interno**: `GridSearchCV` para búsqueda de hiperparámetros.
- **Ciclo externo**: `cross_validate` para evaluación imparcial del modelo.

Devuelve un diccionario con:

- `metricas_evaluacion`: resumen de todas las métricas (test/train mean, std, gap) por fold externo.
- `mejores_parametros`: hiperparámetros óptimos del modelo final.
- `modelo_final`: estimador final re-entrenado en todo el dataset con los mejores parámetros.

#### `permutation_importance_cv(model, X, y, cv, scoring, n_repeats, random_state, model_name)`

Calcula la **importancia por permutación** sobre cada partición de validación cruzada. Promedia las importancias de todos los folds y genera un gráfico de barras horizontales con barras de error (± std), coloreado con la paleta `viridis`.

#### `shap_importance_models(model, X, model_name)`

Calcula la importancia global de variables mediante **SHAP** (`|SHAP value|` promedio), compatible con:

- `RandomForestClassifier` → `TreeExplainer`
- `LogisticRegression` → `LinearExplainer`
- SVM (RBF/Lineal) → `KernelExplainer` con muestreo de fondo (30 instancias)

Genera un gráfico de barras horizontales y devuelve `(df_shap, fig)`.

#### `explicar_clasificacion_individual(model, X, idx_individuo, model_name, clase_idx)`

Genera una **explicación individual SHAP** (waterfall plot) para un paciente específico (`idx_individuo`) y una clase objetivo (`clase_idx`). Compatible con Pipelines, Random Forest, Regresión Logística y SVM. Las explicaciones de modelos probabilísticos se expresan en escala de probabilidad.

### API pública del módulo (`src/modeling/__init__.py`)

```python
from src.modeling import (
    seed,
    scoring,
    cv_repeated_kfold,
    cv_kfold,
    train_model,
    get_confusion_matrix,
    print_scores,
    compare_datasets_wilcoxon,
    graficar_curvas_roc_multiclase,
    nested_cross_validation_multi
)
```

---

## Modelos de Machine Learning (`models/`)

La carpeta `models/` contiene los clasificadores entrenados y serializados con `joblib`. Cada archivo `.joblib` corresponde a un modelo entrenado para la tarea de clasificación triclase del deterioro cognitivo: **Control (0)**, **DCL (1)**, **Demencia (2)**.

| Archivo                                         | Algoritmo           | Descripción                                                                                                                                                 |
| ----------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modelo_logistic_regression_diagnostico.joblib` | Regresión Logística | Clasificador lineal de línea base con regularización L2. Sirve como referencia interpretable y de bajo costo computacional.                                 |
| `modelo_random_forest_diagnostico.joblib`       | Random Forest       | Conjunto de árboles de decisión con bootstrapping. Captura relaciones no lineales y permite análisis de importancia de variables (SHAP Tree + Permutación). |
| `modelo_svm_lineal_diagnostico.joblib`          | SVM Lineal          | Máquina de Vectores de Soporte con kernel lineal. Efectivo en espacios de alta dimensión y datasets de tamaño moderado.                                     |
| `modelo_svm_rbf_diagnostico.joblib`             | SVM RBF             | Máquina de Vectores de Soporte con kernel radial (RBF). Modela fronteras de decisión no lineales en el espacio de características cognitivas.               |

### Cargar un modelo serializado

```python
import joblib

modelo = joblib.load("models/modelo_random_forest_diagnostico.joblib")
y_pred = modelo.predict(X_new)
```

> **Nota**: Los modelos fueron serializados como **Pipelines completos** de scikit-learn (imputación + escalado + clasificador), por lo que pueden recibir directamente un `DataFrame` con las features originales sin preprocesamiento adicional.

---

## Notebooks

Los notebooks están organizados por etapa del proyecto CRISP-DM dentro de la carpeta `notebooks/`:

### `notebooks/eda/` — Análisis Exploratorio de Datos

| Notebook          | Descripción                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `dc_eda_v1.ipynb` | EDA completo: distribuciones, correlaciones, análisis por dominio cognitivo y visualizaciones de la muestra de pacientes. |
| `dc_eda_v2.ipynb` | EDA refinado y complementario con visualizaciones adicionales y foco en el desbalance de clases.                          |

### `notebooks/pipeline/` — Construcción del Pipeline de Datos

| Notebook               | Descripción                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `dc_pipeline_v1.ipynb` | Primera versión del pipeline ETL: exploración de la fuente de datos Excel y estructuración inicial.                      |
| `dc_pipeline_v2.ipynb` | Pipeline completo y modularizado: integración con `src/pipeline`, construcción de `df_complete` y validación de salidas. |

### `notebooks/modeling/` — Modelado e Interpretabilidad

| Notebook                    | Descripción                                                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dc_ia_v1.ipynb`            | Primera iteración de modelado: entrenamiento y comparación inicial de clasificadores (Regresión Logística, Random Forest, SVM Lineal y SVM RBF) con validación cruzada estratificada.  |
| `dc_ia_v2.ipynb`            | Segunda iteración de modelado: ajuste de hiperparámetros con Nested Cross-Validation, análisis de overfitting y selección del modelo final. Serialización de los modelos en `models/`. |
| `dc_wilcoxon.ipynb`         | Comparación estadística entre datasets construidos con la media y la mediana mediante la prueba de Wilcoxon de rangos con signo.                                                       |
| `dc_red_dim.ipynb`          | Reducción de dimensionalidad: exploración de PCA y técnicas similares para visualización del espacio de características cognitivas.                                                    |
| `dc_interpretability.ipynb` | Interpretabilidad de los modelos entrenados: importancia por permutación, valores SHAP globales e individuales (waterfall plots) y análisis de contribución de variables clínicas.     |

---

## Notas

- El pipeline conserva pacientes y evita eliminar filas por nulos.
- La imputación se hace por grupo clínico `dc` cuando la tasa de nulos lo permite.
- `df_complete` está pensado para alimentar gráficas, estadística inferencial y modelos de clasificación.
- Los modelos en `models/` son Pipelines completos y no requieren preprocesamiento externo al usarlos para inferencia.
- La semilla global `19971711` garantiza la reproducibilidad de todos los experimentos de modelado.
