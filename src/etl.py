"""
etl.py — Pipeline ETL para el procesamiento de datos neuropsicológicos.

Pasos principales
-----------------
1. Extracción   : Descarga el archivo Excel desde GitHub.
2. Limpieza     : Elimina filas vacías y estandariza encabezados.
3. Detección    : Identifica el formato de cada hoja (Tabla 0 / Tabla 1).
4. Extracción   : Obtiene los features neuropsicológicos de cada hoja.
5. Normalización: Estandariza los valores categóricos (interpretación clínica).
6. Imputación   : Imputa nulos con mediana/moda por grupo clínico.
7. Integración  : Construye `df_complete` con dominios cognitivos agregados.

Retorno de `run_etl()`
----------------------
Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (df_tabla_0_imp, df_tabla_1_imp, df_complete)
"""

from __future__ import annotations

import unicodedata
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values
from io import BytesIO

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes de dominio — columnas de identificación de cada paciente
# ---------------------------------------------------------------------------

ID_COLS: List[str] = ["dc", "age", "nivel_estudio", "sheet_name"]

# Features extraídos de la Tabla 0 (escolaridad baja)
FEATURES_TABLA_0: List[str] = [
    # Orientación
    "tiempo",
    "persona",
    "espacio",
    # Atención
    "atencion_sostenida_auditiva",
    "atencion_sostenida_visual",
    "atencion_selectiva_visual",
    "atencion_dividida_visual",
    # Lenguaje
    "denominacion",
    "material_verbal_complejo",
    "comprension_de_ordenes",
    # Memoria verbal — California Verbal Learning Test / Verbal del Rey
    "evocacion_inmediata_lista_a",
    "recuerdo_inmediato_lista_a",
    "recuerdo_inmediato_lista_b",
    "recuerdo_libre_a_corto_plazo",
    "recuerdo_libre_a_largo_plazo",
    "reconocimiento",
    # Memoria visual
    "evocacion_diferida",
    # Gnosias / capacidad visuoperceptiva
    "imagenes_sobrepuestas",
    "matrices",
    # Praxis
    "copia_de_figura",
    # Funciones ejecutivas
    "memoria_de_trabajo_digitos_inversos",
    "memoria_de_trabajo_digitos_secuenciales",
    "fluidez_verbal_semantica",
    "semejanzas",
    "matrices",
    "comprension_abstraccion",
    "stroop_palabra",
    "stroop_color",
    "stroop_interferencia",
]

# Features extraídos de la Tabla 1 (escolaridad alta)
FEATURES_TABLA_1: List[str] = [
    # Orientación
    "tiempo",
    "persona",
    "espacio",
    # Atención
    "digitos_en_progresion",
    "deteccion_visual",
    "series_sucesivas",
    # Lenguaje
    "denominacion",
    "semejanzas",
    "material_verbal_complejo",
    "comprension_de_ordenes",
    # Memoria verbal
    "curva_de_memoria_volumen_promedio",
    "memoria_verbal_espontanea_total",
    "memoria_verbal_claves_total",
    "memoria_verbal_reconocimiento",
    "memoria_logica_promedio_historias",
    # Memoria visual
    "caras_codificacion",
    "reconocimiento_caras",
    "evocacion_figura_semicompleja",
    # Gnosias
    "imagenes_sobrepuestas",
    # Praxis
    "copia_de_figura",
    "gestos_simbolicos",
    # Funciones ejecutivas
    "semejanzas",
    "fluidez_verbal_semantica",
    "fluidez_verbal_fonologica",
    "fluidez_no_verbal",
    "retencion_digitos_regresion",
]

# Mapeado de columnas por dominio cognitivo
DOMINIOS: Dict[str, List[str]] = {
    "orientacion": ["tiempo", "persona", "espacio"],
    "atencion": [
        "atencion_sostenida_auditiva",
        "atencion_sostenida_visual",
        "atencion_selectiva_visual",
        "atencion_dividida_visual",
        "digitos_en_progresion",
        "deteccion_visual",
        "series_sucesivas",
    ],
    "lenguaje": [
        "denominacion",
        "comprension_de_ordenes",
        "material_verbal_complejo",
        "semejanzas",
        "comprension_ejecucion_de_ordenes",
    ],
    "memoria_verbal": [
        "evocacion_inmediata_lista_a",
        "recuerdo_inmediato_lista_a",
        "recuerdo_inmediato_lista_b",
        "recuerdo_libre_a_corto_plazo",
        "recuerdo_libre_a_largo_plazo",
        "reconocimiento",
        "curva_de_memoria_volumen_promedio",
        "memoria_verbal_espontanea_total",
        "memoria_verbal_claves_total",
        "memoria_verbal_reconocimiento",
        "memoria_logica_promedio_historias",
    ],
    "memoria_visual": [
        "evocacion_diferida",
        "caras_codificacion",
        "reconocimiento_caras",
        "evocacion_figura_semicompleja",
    ],
    "gnosias": ["imagenes_sobrepuestas", "matrices"],
    "praxis": [
        "copia_de_figura",
        "gestos_simbolicos",
        "imitacion_de_posturas",
    ],
    "ejecutivas": [
        "memoria_de_trabajo_digitos_inversos",
        "memoria_de_trabajo_digitos_secuenciales",
        "evocacion_categorial_semantica",
        "matrices",
        "comprension_abstraccion",
        "stroop_palabra",
        "stroop_colores",
        "stroop_interferencia",
        "fluidez_verbal_semantica",
        "fluidez_verbal_fonologica",
        "fluidez_no_verbal",
        "retencion_digitos_regresion",
    ],
}

# Etiquetas y paleta para visualizaciones (disponibles al importar el módulo)
DC_LABELS: Dict[int, str] = {0: "Control", 1: "DCL", 2: "Demencia"}
DC_ORDER: List[int] = [0, 1, 2]
EDUCATION_LEVEL: Dict[int, str] = {
    0: "Escolaridad Baja",
    1: "Escolaridad Alta",
}

# ---------------------------------------------------------------------------
# 1. Extracción — descarga del archivo Excel
# ---------------------------------------------------------------------------


def load_excel_from_github(dotenv_path: str = ".env") -> Dict[str, pd.DataFrame]:
    """
    Descarga el archivo Excel desde GitHub usando las credenciales del .env.

    Parámetros
    ----------
    dotenv_path : str
        Ruta al archivo .env que contiene DATA_FILE_URL y GH_TOKEN.

    Retorna
    -------
    Dict[str, pd.DataFrame]
        Diccionario {nombre_hoja: DataFrame} con todas las hojas del libro.
    """
    config = dotenv_values(dotenv_path=dotenv_path)
    url = config["DATA_FILE_URL"]
    headers = {"Authorization": f"Bearer {config['GH_TOKEN']}"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    return pd.read_excel(
        BytesIO(response.content),
        header=None,
        sheet_name=None,
        engine="openpyxl",
    )


# ---------------------------------------------------------------------------
# 2. Limpieza — estandarización de hojas
# ---------------------------------------------------------------------------


def clean_sheets(
    xlsx_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Elimina filas completamente vacías de cada hoja y resetea el índice.

    Parámetros
    ----------
    xlsx_data : Dict[str, pd.DataFrame]
        Hojas crudas provenientes de `pd.read_excel`.

    Retorna
    -------
    Dict[str, pd.DataFrame]
        Hojas limpias.
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    for name, sheet in xlsx_data.items():
        sheet = sheet.dropna(how="all").reset_index(drop=True)
        cleaned[name] = sheet
    return cleaned


def clean_value(cadena: str) -> str:
    """
    Estandariza un string: minúsculas, sin espacios laterales, guiones bajos
    en lugar de espacios, sin tildes ni caracteres especiales.

    Parámetros
    ----------
    cadena : str

    Retorna
    -------
    str
        Cadena normalizada.
    """
    return (
        str(cadena)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


# ---------------------------------------------------------------------------
# 3. Detección — formato de tabla por hoja
# ---------------------------------------------------------------------------


def detect_table_format(
    sheet: pd.DataFrame,
    keywords: List[str] = ["espacio"],
    check_columns: List[int] = [0, 1],
) -> int | str:
    """
    Detecta el formato de la tabla buscando palabras clave en columnas clave.

    Parámetros
    ----------
    sheet : pd.DataFrame
        Hoja a analizar.
    keywords : List[str]
        Palabras clave a buscar (por defecto ['espacio']).
    check_columns : List[int]
        Índices de columnas a revisar (0 = primera, 1 = segunda).

    Retorna
    -------
    int | str
        0 si la palabra clave está en la columna 0 (Tabla 0, escolaridad baja).
        1 si está en la columna 1 (Tabla 1, escolaridad alta).
        'no determinada' si no se encontró en ninguna columna.
    """
    for col_idx in check_columns:
        if col_idx >= len(sheet.columns):
            continue

        col_data = sheet.iloc[:, col_idx].astype(str).str.lower().str.strip()
        col_data_clean = col_data.str.replace(" ", "_", regex=False)
        col_data_clean = col_data_clean.str.replace(
            "[áéíóú]",
            lambda m: {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}.get(
                m.group(0), m.group(0)
            ),
            regex=True,
        )
        col_data_clean = col_data_clean.str.replace(r"[\-\.\(\)]", "", regex=True)

        pattern = "|".join(keywords)
        if col_data_clean.str.contains(pattern, na=False, case=False).any():
            return col_idx

    return "no determinada"


# ---------------------------------------------------------------------------
# 4. Extracción — información del paciente y features por hoja
# ---------------------------------------------------------------------------


def extract_patient_info(sheet_name: str) -> Tuple[int | str, str]:
    """
    Extrae la clasificación DC y la edad del nombre de la hoja.

    Parámetros
    ----------
    sheet_name : str
        Nombre de la hoja, e.g. 'S1-F067-58', 'GC1-60', 'F021-72'.

    Retorna
    -------
    Tuple[int | str, str]
        (dc, age)
        dc ∈ {0: Control, 1: DCL (F06), 2: Demencia (F02), 'No determinada'}.
        age: último fragmento tras '-'.
    """
    upper = str(sheet_name).upper()
    if "F06" in upper:
        dc = 1  # Deterioro cognitivo leve
    elif "F02" in upper:
        dc = 2  # Demencia
    elif "GC" in upper:
        dc = 0  # Grupo de control
    else:
        dc = "No determinada"

    age = str(sheet_name).split("-")[-1].strip()
    return dc, age


def extract_features_from_table(
    sheet: pd.DataFrame,
    headers_col: int,
    value_col: int,
    features: List[str],
    headers_clean: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """
    Extrae los valores de los features neuropsicológicos de una hoja.

    Parámetros
    ----------
    sheet : pd.DataFrame
    headers_col : int
        Índice de la columna que contiene los nombres de los tests.
    value_col : int
        Índice de la columna con el puntaje normalizado.
    features : List[str]
        Lista de features (nombres limpios) a extraer.
    headers_clean : pd.Series, opcional
        Serie con los headers ya normalizados. Se calcula si no se pasa.

    Retorna
    -------
    Dict[str, object]
        Diccionario {feature: valor_extraído | None}.
    """
    if headers_clean is None:
        headers_clean = sheet.iloc[:, headers_col].astype(str).apply(clean_value)

    # Índice de la primera fila que contiene cada feature
    feature_to_idx: Dict[str, int] = {}
    for feature in features:
        mask = headers_clean.str.contains(feature, case=False, na=False)
        if mask.any():
            feature_to_idx[feature] = headers_clean[mask].index[0]

    return {
        feature: (
            sheet.iloc[feature_to_idx[feature], value_col]
            if feature in feature_to_idx
            else None
        )
        for feature in features
    }


def search_values(
    data: Dict[str, pd.DataFrame],
    features: List[str],
    type_of_table: Dict[str, int | str],
) -> pd.DataFrame:
    """
    Recorre todas las hojas y extrae los features para cada paciente.

    Parámetros
    ----------
    data : Dict[str, pd.DataFrame]
        Hojas limpias.
    features : List[str]
        Features a extraer.
    type_of_table : Dict[str, int | str]
        Formato detectado por hoja (0, 1 o 'no determinada').

    Retorna
    -------
    pd.DataFrame
        Un DataFrame con una fila por paciente y una columna por feature.
    """
    # Configuración de columnas según formato de tabla
    table_configs: Dict[int, Dict[str, int]] = {
        0: {"headers_col": 0, "value_col": 3, "nivel_estudio": 0},
        1: {"headers_col": 1, "value_col": 4, "nivel_estudio": 1},
    }

    results = []

    for sheet_name, sheet in data.items():
        table_format = type_of_table[sheet_name]

        if table_format == "no determinada":
            continue

        config = table_configs[table_format]
        dc, age = extract_patient_info(sheet_name)

        # Pre-procesar headers una sola vez
        headers_clean = (
            sheet.iloc[:, config["headers_col"]].astype(str).apply(clean_value)
        )

        features_dict = extract_features_from_table(
            sheet,
            config["headers_col"],
            config["value_col"],
            features,
            headers_clean,
        )

        results.append(
            {
                "sheet_name": sheet_name,
                "nivel_estudio": config["nivel_estudio"],
                "dc": dc,
                "age": age,
                **features_dict,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Normalización — estandarización de valores categóricos
# ---------------------------------------------------------------------------


def _limpiar_tildes(texto: str) -> str:
    """Elimina las tildes de un string mediante normalización Unicode NFD."""
    texto_nfd = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in texto_nfd if unicodedata.category(c) != "Mn")


def _limpiar_caracteres_especiales(texto: str) -> str:
    """Conserva solo alfanuméricos, espacios, guiones y guiones bajos."""
    return "".join(
        c for c in str(texto) if c.isalnum() or c in [" ", "_", "-"]
    )


def limpiar_texto(texto: object) -> Optional[str]:
    """
    Limpia un valor de celda: quita tildes, caracteres especiales,
    convierte a minúsculas y strip. Retorna None si el valor es nulo.

    Parámetros
    ----------
    texto : object
        Valor crudo de celda.

    Retorna
    -------
    Optional[str]
    """
    if pd.isna(texto) or str(texto) in ("None", "nan"):
        return None

    texto = str(texto)
    texto = _limpiar_tildes(texto)
    texto = _limpiar_caracteres_especiales(texto)
    texto = texto.lower().strip()

    return texto if texto else None


# Mapa de valores de Tabla 1 → categoría canónica (4 niveles)
_REEMPLAZOS_TABLA_1: Dict[str, str] = {
    "alteracion severa": "alteracion_severa",
    "altearcion severa": "alteracion_severa",
    "alteracion leve": "bajo",
    "alteracion": "bajo",
    "alteracion moderada": "bajo",
    "aleracion leve": "bajo",
    "deficit": "bajo",
    "alteracion leve-moderada": "bajo",
    "alto": "alto",
    "promedio": "promedio",
    "normal alto": "alto",
    "maximo": "alto",
}

# Mapa de valores de Tabla 0 → categoría canónica (4 niveles)
_REEMPLAZOS_TABLA_0: Dict[str, str] = {
    "alto": "alto",
    "deficit": "alteracion_severa",
    "promedio": "promedio",
    "bajo": "bajo",
    "maximo": "alto",
}


def _normalizar_df(df: pd.DataFrame, reemplazos: Dict[str, str]) -> pd.DataFrame:
    """
    Aplica `limpiar_texto` y luego los reemplazos categóricos a todas las
    columnas que no son de identificación.

    Parámetros
    ----------
    df : pd.DataFrame
    reemplazos : Dict[str, str]
        Mapa de valor limpio → valor canónico.

    Retorna
    -------
    pd.DataFrame
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    for col in feature_cols:
        df[col] = df[col].apply(limpiar_texto)
        df[col] = df[col].replace(reemplazos)

    return df


# ---------------------------------------------------------------------------
# 6. Imputación clínica
# ---------------------------------------------------------------------------


def null_data_info(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """
    Resumen de valores nulos por columna (excluye columnas de ID).

    Parámetros
    ----------
    df : pd.DataFrame
    id_cols : List[str]

    Retorna
    -------
    pd.DataFrame
        Columnas: nulos, %_nulos, dtype, n_unicos.
    """
    cols = [c for c in df.columns if c not in id_cols]
    return pd.DataFrame(
        {
            "nulos": df[cols].isna().sum(),
            "%_nulos": (df[cols].isna().mean() * 100).round(2),
            "dtype": df[cols].dtypes.astype(str),
            "n_unicos": df[cols].nunique(dropna=True),
        }
    ).sort_values("%_nulos", ascending=False)


def imputacion_null(
    df: pd.DataFrame,
    id_cols: List[str],
    group_col: str = "dc",
    high_missing: float = 0.30,
) -> pd.DataFrame:
    """
    Imputa valores nulos de manera clínica:

    * Agrega columnas indicadoras `{col}_missing` para toda columna con nulos.
    * Columnas con tasa de nulos ≥ `high_missing` NO se imputan (demasiados nulos).
    * Categóricas  → moda por grupo clínico; fallback: moda global.
    * Numéricas    → mediana por grupo clínico; fallback: mediana global.

    Parámetros
    ----------
    df : pd.DataFrame
    id_cols : List[str]
        Columnas de identificación que se excluyen de la imputación.
    group_col : str
        Columna de agrupación clínica (por defecto 'dc').
    high_missing : float
        Umbral de tasa de nulos a partir del cual no se imputa (por defecto 0.30).

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con los nulos imputados.
    """
    df_imp = df.copy()
    cols = [col for col in df.columns if col not in id_cols]
    missing_rate = df[cols].isna().mean()

    for col in cols:
        porcent = missing_rate[col]
        if porcent == 0:
            continue

        # Columna indicadora de ausencia
        df_imp[f"{col}_missing"] = df[col].isna().astype(int)

        # No imputar si la tasa supera el umbral
        if porcent >= high_missing:
            continue

        # Detectar dinámicamente si es numérica intentando calcular la mediana
        is_numeric = False
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Intentamos calcular una mediana de prueba para asegurar que no tire error
                _ = df[col].median()
                is_numeric = True
            except Exception:
                is_numeric = False

        if is_numeric:
            # Imputación numérica: mediana por grupo clínico
            df_imp[col] = df.groupby(group_col)[col].transform(
                lambda s: s.fillna(s.median())
            )

            # Fallback global
            if df_imp[col].isna().any():
                df_imp[col] = df_imp[col].fillna(df[col].median())
        else:
            # Imputación categórica: moda por grupo clínico
            def _fill_mode(s: pd.Series) -> pd.Series:
                mode = s.mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) > 0 else np.nan
                return s.fillna(fill_value)

            df_imp[col] = df.groupby(group_col)[col].transform(_fill_mode)

            # Fallback global
            if df_imp[col].isna().any():
                global_mode = df[col].mode(dropna=True)
                if len(global_mode) > 0:
                    df_imp[col] = df_imp[col].fillna(global_mode.iloc[0])

    return df_imp


# ---------------------------------------------------------------------------
# 7. Construcción de df_complete con dominios cognitivos
# ---------------------------------------------------------------------------

# Mapa ordinal para las 4 categorías clínicas
_ORDINAL_MAP: Dict[str, int] = {
    "alteracion_severa": 1,
    "bajo": 2,
    "promedio": 3,
    "alto": 4,
}


def _to_ordinal(series: pd.Series) -> pd.Series:
    """Convierte una columna categórica al mapa ordinal de 4 niveles."""
    return series.map(_ORDINAL_MAP)


def _domain_score(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Calcula el puntaje promedio ordinal de un dominio cognitivo.

    Solo utiliza las columnas presentes en `df`. Si ninguna columna
    del dominio está presente, retorna una Serie de None.

    Parámetros
    ----------
    df : pd.DataFrame
    cols : List[str]
        Columnas del dominio.

    Retorna
    -------
    pd.Series
    """
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return pd.Series([None] * len(df), index=df.index)
    scores = df[cols_present].apply(_to_ordinal)
    return scores.mean(axis=1)


def build_df_complete(
    df_tabla_0_imp: pd.DataFrame,
    df_tabla_1_imp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el DataFrame integrado con los dominios cognitivos agregados.

    Parámetros
    ----------
    df_tabla_0_imp : pd.DataFrame
        Tabla 0 ya imputada.
    df_tabla_1_imp : pd.DataFrame
        Tabla 1 ya imputada.

    Retorna
    -------
    pd.DataFrame
        `df_complete` con columnas: sheet_name, nivel_estudio, dc, age,
        age_num, y una columna numérica por cada dominio cognitivo.
        Sin valores nulos (los dominios con nulos moderados se imputan por
        mediana del grupo clínico).
    """
    cols_id = ["sheet_name", "nivel_estudio", "dc", "age"]

    # Unión de las dos tablas imputadas
    df_union = pd.concat(
        [df_tabla_0_imp, df_tabla_1_imp], ignore_index=True, sort=False
    )

    # Construcción de los dominios cognitivos (puntaje promedio ordinal)
    df_complete = df_union[cols_id].copy()
    for dominio, cols in DOMINIOS.items():
        df_complete[dominio] = _domain_score(df_union, cols)

    # Imputación final de nulos en dominios (mediana por grupo clínico)
    dominio_cols = list(DOMINIOS.keys())
    for col in dominio_cols:
        if df_complete[col].isna().mean() < 0.30:
            df_complete[col] = df_complete.groupby("dc")[col].transform(
                lambda s: s.fillna(s.median())
            )

    # Conversión de tipos
    df_complete["nivel_estudio"] = (
        df_complete["nivel_estudio"].astype(str).str.strip().astype(int)
    )
    df_complete["age"] = df_complete["age"].astype(str).str.strip().astype(int)
    df_complete["dc"] = df_complete["dc"].astype(int)

    # Columna numérica de edad (útil para gráficas)
    df_complete["age_num"] = pd.to_numeric(df_complete["age"], errors="coerce")

    return df_complete


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def run_etl(
    dotenv_path: str = ".env",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline ETL completo.

    Pasos
    -----
    1. Descarga el Excel desde GitHub.
    2. Limpia las hojas.
    3. Detecta el formato de cada hoja (Tabla 0 / Tabla 1).
    4. Extrae los features de cada hoja por separado.
    5. Normaliza los valores categóricos (interpretación clínica).
    6. Imputa los valores nulos con criterio clínico.
    7. Construye `df_complete` con dominios cognitivos.

    Parámetros
    ----------
    dotenv_path : str
        Ruta al archivo .env. Por defecto '.env'.
    verbose : bool
        Si True, imprime resúmenes de cada etapa.

    Retorna
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_tabla_0_imp, df_tabla_1_imp, df_complete)
    """
    # ------------------------------------------------------------------
    # 1. Extracción
    # ------------------------------------------------------------------
    if verbose:
        print("Descargando datos desde GitHub...")
    xlsx_data = load_excel_from_github(dotenv_path)

    # ------------------------------------------------------------------
    # 2. Limpieza de hojas
    # ------------------------------------------------------------------
    cleaned_sheets = clean_sheets(xlsx_data)
    if verbose:
        print(f"Hojas cargadas: {len(cleaned_sheets)}")

    # ------------------------------------------------------------------
    # 3. Detección de formato
    # ------------------------------------------------------------------
    type_of_table: Dict[str, int | str] = {
        name: detect_table_format(sheet)
        for name, sheet in cleaned_sheets.items()
    }
    n_tabla_0 = list(type_of_table.values()).count(0)
    n_tabla_1 = list(type_of_table.values()).count(1)
    n_indet = list(type_of_table.values()).count("no determinada")
    if verbose:
        print(
            f"Tabla 0 (escolaridad baja) : {n_tabla_0}\n"
            f"Tabla 1 (escolaridad alta) : {n_tabla_1}\n"
            f"No determinadas            : {n_indet}"
        )

    # ------------------------------------------------------------------
    # 4. Extracción de features por tabla
    # ------------------------------------------------------------------
    cleaned_tabla_0 = {
        name: sheet
        for name, sheet in cleaned_sheets.items()
        if type_of_table[name] == 0
    }
    cleaned_tabla_1 = {
        name: sheet
        for name, sheet in cleaned_sheets.items()
        if type_of_table[name] == 1
    }

    df_tabla_0 = search_values(cleaned_tabla_0, FEATURES_TABLA_0, type_of_table)
    df_tabla_1 = search_values(cleaned_tabla_1, FEATURES_TABLA_1, type_of_table)

    if verbose:
        print(
            f"df_tabla_0 crudo : {df_tabla_0.shape}\n"
            f"df_tabla_1 crudo : {df_tabla_1.shape}"
        )

    # ------------------------------------------------------------------
    # 5. Normalización de valores categóricos
    # ------------------------------------------------------------------
    df_tabla_0 = _normalizar_df(df_tabla_0, _REEMPLAZOS_TABLA_0)
    df_tabla_1 = _normalizar_df(df_tabla_1, _REEMPLAZOS_TABLA_1)

    # ------------------------------------------------------------------
    # 6. Imputación clínica
    # ------------------------------------------------------------------
    if verbose:
        print("\n🔧  Imputando nulos...")
        print("--- Perfil nulos Tabla 0 ---")
        perfil_t0 = null_data_info(df_tabla_0, ID_COLS)
        print(perfil_t0[perfil_t0["nulos"] > 0])
        print("\n--- Perfil nulos Tabla 1 ---")
        perfil_t1 = null_data_info(df_tabla_1, ID_COLS)
        print(perfil_t1[perfil_t1["nulos"] > 0])

    df_tabla_0_imp = imputacion_null(df_tabla_0, ID_COLS)
    df_tabla_1_imp = imputacion_null(df_tabla_1, ID_COLS)

    if verbose:
        print(
            f"\ndf_tabla_0_imp : {df_tabla_0_imp.shape}\n"
            f"df_tabla_1_imp : {df_tabla_1_imp.shape}"
        )

    # ------------------------------------------------------------------
    # 7. Construcción de df_complete
    # ------------------------------------------------------------------
    df_complete = build_df_complete(df_tabla_0_imp, df_tabla_1_imp)

    if verbose:
        print(f"\n🏁  df_complete listo: {df_complete.shape}")
        print(f"    Nulos restantes:\n{df_complete.isnull().sum()}")
        n_ctrl = (df_complete["dc"] == 0).sum()
        n_dcl = (df_complete["dc"] == 1).sum()
        n_dem = (df_complete["dc"] == 2).sum()
        print(
            f"\n    Distribución clínica:\n"
            f"      Control  : {n_ctrl}\n"
            f"      DCL      : {n_dcl}\n"
            f"      Demencia : {n_dem}\n"
            f"      Total    : {len(df_complete)}"
        )

    return df_tabla_0_imp, df_tabla_1_imp, df_complete


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_tabla_0_imp, df_tabla_1_imp, df_complete = run_etl(verbose=True)
    print("\nPrimeras filas de df_complete:")
    print(df_complete.head())
