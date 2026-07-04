"""ETL notebook-derived pipeline for the dc_model_ts project.

This module mirrors the notebook workflow and keeps the three main outputs
requested by the user:

* df_tabla_0_imp
* df_tabla_1_imp
* df_complete

The implementation favors small composable helpers, explicit constants and a
single public entrypoint: run_etl_notebook().
"""

from __future__ import annotations

import unicodedata
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

warnings.filterwarnings("ignore")

DEFAULT_DOTENV_PATH = ".env"

ID_COLS: List[str] = ["dc", "age", "nivel_estudio", "sheet_name"]

DC_LABELS: Dict[int, str] = {0: "Control", 1: "DCL", 2: "Demencia"}
DC_PALETTE: Dict[int, str] = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
DC_ORDER: List[int] = [0, 1, 2]
EDUCATION_LEVEL: Dict[int, str] = {0: "Escolaridad Baja", 1: "Escolaridad Alta"}

FEATURES_TABLA_0: List[str] = [
    "tiempo",
    "persona",
    "espacio",
    "atencion_sostenida_auditiva",
    "atencion_sostenida_visual",
    "atencion_selectiva_visual",
    "atencion_dividida_visual",
    "denominacion",
    "material_verbal_complejo",
    "comprension_de_ordenes",
    "evocacion_inmediata_lista_a",
    "recuerdo_inmediato_lista_a",
    "recuerdo_inmediato_lista_b",
    "recuerdo_libre_a_corto_plazo",
    "recuerdo_libre_a_largo_plazo",
    "reconocimiento",
    "evocacion_diferida",
    "imagenes_sobrepuestas",
    "matrices",
    "copia_de_figura",
    "memoria_de_trabajo_digitos_inversos",
    "memoria_de_trabajo_digitos_secuenciales",
    "fluidez_verbal_semantica",
    "semejanzas",
    "comprension_abstraccion",
    "stroop_palabra",
    "stroop_color",
    "stroop_interferencia",
]

FEATURES_TABLA_1: List[str] = [
    "tiempo",
    "persona",
    "espacio",
    "digitos_en_progresion",
    "deteccion_visual",
    "series_sucesivas",
    "denominacion",
    "semejanzas",
    "material_verbal_complejo",
    "comprension_de_ordenes",
    "curva_de_memoria_volumen_promedio",
    "memoria_verbal_espontanea_total",
    "memoria_verbal_claves_total",
    "memoria_verbal_reconocimiento",
    "memoria_logica_promedio_historias",
    "caras_codificacion",
    "reconocimiento_caras",
    "evocacion_figura_semicompleja",
    "imagenes_sobrepuestas",
    "copia_de_figura",
    "gestos_simbolicos",
    "fluidez_verbal_semantica",
    "fluidez_verbal_fonologica",
    "fluidez_no_verbal",
    "retencion_digitos_regresion",
]

FEATURES_TABLAS: List[str] = [
    "tiempo",
    "persona",
    "espacio",
    "atencion_sostenida_auditiva",
    "atencion_dividida_visual",
    "digitos_en_progresion",
    "deteccion_visual",
    "denominacion",
    "material_verbal_complejo",
    "semejanzas",
    "evocacion_diferida",
    "evocacion_figura_semicompleja",
    "constructiva",
    "copia_de_figura_compleja",
    "memoria_de_trabajo_digitos_inversos",
    "comprension_abstraccion",
    "comprension",
    "retencion_digitos_regresion",
]

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

_REEMPLAZOS_TABLA_0: Dict[str, str] = {
    "alto": "alto",
    "deficit": "alteracion_severa",
    "promedio": "promedio",
    "bajo": "bajo",
    "maximo": "alto",
}

_CROSS_TABLE_ALIASES: Dict[str, str] = {
    "atencion_sostenida_auditiva": "digitos_en_progresion",
    "atencion_dividida_visual": "deteccion_visual",
    "evocacion_diferida": "evocacion_figura_semicompleja",
    "constructiva": "copia_de_figura_compleja",
    "memoria_de_trabajo_digitos_inversos": "retencion_digitos_regresion",
    "comprension_abstraccion": "comprension",
}

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
    "praxis": ["copia_de_figura", "gestos_simbolicos", "imitacion_de_posturas"],
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


def load_excel_from_github(dotenv_path: str = DEFAULT_DOTENV_PATH) -> Dict[str, pd.DataFrame]:
    """Load the source workbook from GitHub using environment credentials."""
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


def clean_sheets(xlsx_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Drop fully empty rows and reset the index for every sheet."""
    return {
        name: sheet.dropna(how="all").reset_index(drop=True)
        for name, sheet in xlsx_data.items()
    }


def clean_value(value: object) -> str:
    """Normalize a label for robust text matching."""
    normalized = (
        str(value)
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
    return normalized


def detect_table_format(
    sheet: pd.DataFrame,
    keywords: Sequence[str] = ("espacio",),
    check_columns: Sequence[int] = (0, 1),
) -> int | str:
    """Detect whether a sheet follows table format 0 or 1."""
    for col_idx in check_columns:
        if col_idx >= len(sheet.columns):
            continue

        col_data = sheet.iloc[:, col_idx].astype(str).str.lower().str.strip()
        col_data_clean = col_data.str.replace(" ", "_", regex=False)
        col_data_clean = col_data_clean.str.replace(
            "[áéíóú]",
            lambda match: {
                "á": "a",
                "é": "e",
                "í": "i",
                "ó": "o",
                "ú": "u",
            }.get(match.group(0), match.group(0)),
            regex=True,
        )
        col_data_clean = col_data_clean.str.replace(r"[\-\.\(\)]", "", regex=True)

        pattern = "|".join(keywords)
        if col_data_clean.str.contains(pattern, na=False, case=False).any():
            return col_idx

    return "no determinada"


def extract_patient_info(sheet_name: str) -> Tuple[int | str, str]:
    """Extract dc class and age from the worksheet name."""
    upper = str(sheet_name).upper()
    if "F06" in upper:
        dc: int | str = 1
    elif "F02" in upper:
        dc = 2
    elif "GC" in upper:
        dc = 0
    else:
        dc = "No determinada"

    age = str(sheet_name).split("-")[-1].strip()
    return dc, age


def extract_features_from_table(
    sheet: pd.DataFrame,
    headers_col: int,
    value_col: int,
    features: Sequence[str],
    headers_clean: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """Extract feature values from a single sheet."""
    if headers_clean is None:
        headers_clean = sheet.iloc[:, headers_col].astype(str).apply(clean_value)

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
    features: Sequence[str],
    type_of_table: Dict[str, int | str],
) -> pd.DataFrame:
    """Extract all requested features from the provided sheets."""
    table_configs: Dict[int, Dict[str, int]] = {
        0: {"headers_col": 0, "value_col": 3, "nivel_estudio": 0},
        1: {"headers_col": 1, "value_col": 4, "nivel_estudio": 1},
    }

    results: List[Dict[str, object]] = []
    for sheet_name, sheet in data.items():
        table_format = type_of_table[sheet_name]
        if table_format == "no determinada":
            continue

        config = table_configs[table_format]
        dc, age = extract_patient_info(sheet_name)
        headers_clean = sheet.iloc[:, config["headers_col"]].astype(str).apply(clean_value)

        features_dict = extract_features_from_table(
            sheet=sheet,
            headers_col=config["headers_col"],
            value_col=config["value_col"],
            features=features,
            headers_clean=headers_clean,
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


def _limpiar_tildes(texto: object) -> Optional[str]:
    texto_nfd = unicodedata.normalize("NFD", str(texto))
    cleaned = "".join(c for c in texto_nfd if unicodedata.category(c) != "Mn")
    cleaned = cleaned.strip()
    return cleaned or None


def _limpiar_caracteres_especiales(texto: object) -> str:
    return "".join(c for c in str(texto) if c.isalnum() or c in [" ", "_", "-"])


def limpiar_texto(texto: object) -> Optional[str]:
    """Clean categorical values while preserving None-like values."""
    if pd.isna(texto) or str(texto) in ("None", "nan"):
        return None

    texto_limpio = _limpiar_tildes(texto)
    if texto_limpio is None:
        return None

    texto_limpio = _limpiar_caracteres_especiales(texto_limpio)
    texto_limpio = texto_limpio.lower().strip()
    return texto_limpio or None


def _normalizar_df(df: pd.DataFrame, reemplazos: Dict[str, str]) -> pd.DataFrame:
    """Normalize non-ID columns with the notebook replacement rules."""
    df_norm = df.copy()
    feature_cols = [col for col in df_norm.columns if col not in ID_COLS]

    for col in feature_cols:
        df_norm[col] = df_norm[col].apply(limpiar_texto)
        df_norm[col] = df_norm[col].replace(reemplazos)

    return df_norm


def normalize_tabla_0(df: pd.DataFrame) -> pd.DataFrame:
    return _normalizar_df(df, _REEMPLAZOS_TABLA_0)


def normalize_tabla_1(df: pd.DataFrame) -> pd.DataFrame:
    return _normalizar_df(df, _REEMPLAZOS_TABLA_1)


def null_data_info(df: pd.DataFrame, id_cols: Sequence[str]) -> pd.DataFrame:
    """Return a compact null summary excluding ID columns."""
    cols = [col for col in df.columns if col not in id_cols]
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
    id_cols: Sequence[str],
    group_col: str = "dc",
    high_missing: float = 0.30,
) -> pd.DataFrame:
    """Impute missing values using the notebook's clinical logic."""
    df_imp = df.copy()
    cols = [col for col in df.columns if col not in id_cols]
    missing_rate = df[cols].isna().mean()

    for col in cols:
        porcent = missing_rate[col]
        if porcent == 0:
            continue

        df_imp[f"{col}_missing"] = df[col].isna().astype(int)

        if porcent >= high_missing:
            continue

        if df[col].dtype == "object":

            def _fill_mode(series: pd.Series) -> pd.Series:
                mode = series.mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) > 0 else np.nan
                return series.fillna(fill_value)

            df_imp[col] = df.groupby(group_col)[col].transform(_fill_mode)
            if df_imp[col].isna().any():
                global_mode = df[col].mode(dropna=True)
                if len(global_mode) > 0:
                    df_imp[col] = df_imp[col].fillna(global_mode.iloc[0])
        else:
            df_imp[col] = df.groupby(group_col)[col].transform(
                lambda series: series.fillna(series.median())
            )
            if df_imp[col].isna().any():
                df_imp[col] = df_imp[col].fillna(df[col].median())

    return df_imp


_ORDINAL_MAP: Dict[str, int] = {
    "alteracion_severa": 1,
    "bajo": 2,
    "promedio": 3,
    "alto": 4,
}


def _to_ordinal(series: pd.Series) -> pd.Series:
    return series.map(_ORDINAL_MAP)


def _domain_score(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    cols_present = [col for col in cols if col in df.columns]
    if not cols_present:
        return pd.Series([None] * len(df), index=df.index)

    scores = df[cols_present].apply(_to_ordinal)
    return scores.mean(axis=1)


def _harmonize_union_for_complete(df_union: pd.DataFrame) -> pd.DataFrame:
    """Apply the cross-table fallbacks used by the notebook."""
    df_harmonized = df_union.copy()
    for target_col, source_col in _CROSS_TABLE_ALIASES.items():
        if target_col not in df_harmonized.columns:
            df_harmonized[target_col] = df_harmonized.get(source_col)
        elif source_col in df_harmonized.columns:
            df_harmonized[target_col] = df_harmonized[target_col].fillna(
                df_harmonized[source_col]
            )

    drop_candidates = [
        col for col in _CROSS_TABLE_ALIASES.values() if col in df_harmonized.columns
    ]
    if drop_candidates:
        df_harmonized = df_harmonized.drop(columns=drop_candidates)

    return df_harmonized


def build_df_complete(
    df_tabla_0_imp: pd.DataFrame,
    df_tabla_1_imp: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final analysis dataframe with cognitive domain scores."""
    cols_id = ["sheet_name", "nivel_estudio", "dc", "age"]

    df_union = pd.concat(
        [df_tabla_0_imp, df_tabla_1_imp],
        ignore_index=True,
        sort=False,
    )
    df_union = _harmonize_union_for_complete(df_union)

    df_complete = df_union[cols_id].copy()
    for dominio, cols in DOMINIOS.items():
        df_complete[dominio] = _domain_score(df_union, cols)

    for col in DOMINIOS:
        if df_complete[col].isna().mean() < 0.30:
            df_complete[col] = df_complete.groupby("dc")[col].transform(
                lambda series: series.fillna(series.median())
            )
        if df_complete[col].isna().any():
            df_complete[col] = df_complete[col].fillna(df_complete[col].median())

    df_complete["nivel_estudio"] = (
        df_complete["nivel_estudio"].astype(str).str.strip().astype(int)
    )
    df_complete["age"] = df_complete["age"].astype(str).str.strip().astype(int)
    df_complete["dc"] = df_complete["dc"].astype(int)
    df_complete["age_num"] = pd.to_numeric(df_complete["age"], errors="coerce")

    return df_complete


def _build_tables(
    cleaned_sheets: Dict[str, pd.DataFrame],
    type_of_table: Dict[str, int | str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_tabla_0 = {
        name: sheet for name, sheet in cleaned_sheets.items() if type_of_table[name] == 0
    }
    cleaned_tabla_1 = {
        name: sheet for name, sheet in cleaned_sheets.items() if type_of_table[name] == 1
    }
    df_tabla_0 = search_values(cleaned_tabla_0, FEATURES_TABLA_0, type_of_table)
    df_tabla_1 = search_values(cleaned_tabla_1, FEATURES_TABLA_1, type_of_table)
    return df_tabla_0, df_tabla_1


def run_etl_notebook(
    dotenv_path: str = DEFAULT_DOTENV_PATH,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full notebook-derived ETL and return the three outputs."""
    if verbose:
        print("Downloading workbook from GitHub...")
    xlsx_data = load_excel_from_github(dotenv_path)

    cleaned_sheets = clean_sheets(xlsx_data)
    if verbose:
        print(f"Sheets loaded: {len(cleaned_sheets)}")

    type_of_table: Dict[str, int | str] = {
        name: detect_table_format(sheet) for name, sheet in cleaned_sheets.items()
    }

    if verbose:
        print(
            f"Table 0: {list(type_of_table.values()).count(0)}\n"
            f"Table 1: {list(type_of_table.values()).count(1)}\n"
            f"Undetermined: {list(type_of_table.values()).count('no determinada')}"
        )

    df_tabla_0, df_tabla_1 = _build_tables(cleaned_sheets, type_of_table)

    df_tabla_0 = normalize_tabla_0(df_tabla_0)
    df_tabla_1 = normalize_tabla_1(df_tabla_1)

    if verbose:
        print(f"df_tabla_0 raw: {df_tabla_0.shape}")
        print(f"df_tabla_1 raw: {df_tabla_1.shape}")
        print("\nNull profile table 0:")
        print(null_data_info(df_tabla_0, ID_COLS).query("nulos > 0"))
        print("\nNull profile table 1:")
        print(null_data_info(df_tabla_1, ID_COLS).query("nulos > 0"))

    df_tabla_0_imp = imputacion_null(df_tabla_0, ID_COLS)
    df_tabla_1_imp = imputacion_null(df_tabla_1, ID_COLS)

    if verbose:
        print(f"\ndf_tabla_0_imp: {df_tabla_0_imp.shape}")
        print(f"df_tabla_1_imp: {df_tabla_1_imp.shape}")

    df_complete = build_df_complete(df_tabla_0_imp, df_tabla_1_imp)

    if verbose:
        print(f"\ndf_complete ready: {df_complete.shape}")
        print(f"Remaining nulls:\n{df_complete.isnull().sum()}")

    return df_tabla_0_imp, df_tabla_1_imp, df_complete


def run_etl(
    dotenv_path: str = DEFAULT_DOTENV_PATH,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility alias for the new notebook-derived pipeline."""
    return run_etl_notebook(dotenv_path=dotenv_path, verbose=verbose)


__all__ = [
    "DC_LABELS",
    "DC_ORDER",
    "DC_PALETTE",
    "DOMINIOS",
    "EDUCATION_LEVEL",
    "FEATURES_TABLA_0",
    "FEATURES_TABLA_1",
    "FEATURES_TABLAS",
    "ID_COLS",
    "build_df_complete",
    "clean_sheets",
    "clean_value",
    "detect_table_format",
    "extract_features_from_table",
    "extract_patient_info",
    "imputacion_null",
    "load_excel_from_github",
    "null_data_info",
    "normalize_tabla_0",
    "normalize_tabla_1",
    "run_etl",
    "run_etl_notebook",
    "search_values",
]


if __name__ == "__main__":
    df_tabla_0_imp, df_tabla_1_imp, df_complete = run_etl_notebook(verbose=True)
    print("\nFirst rows of df_complete:")
    print(df_complete.head())