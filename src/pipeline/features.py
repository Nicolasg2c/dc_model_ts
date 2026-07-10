"""
features.py — Extracción de features neuropsicológicos por hoja.

Pasos 4 del pipeline ETL:
  - Extrae información del paciente desde el nombre de la hoja.
  - Extrae los puntajes de cada feature neuropsicológico según el formato
    de tabla detectado (Tabla 0 / Tabla 1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import TABLE_CONFIGS
from .utils import clean_value



#Extracción — información del paciente y features por hoja

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


# Extracción — features neuropsicológicos por hoja
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

# Búsqueda — valores de cada dominio cognitivo por hoja
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
    results = []

    for sheet_name, sheet in data.items():
        table_format = type_of_table[sheet_name]

        if table_format == "no determinada":
            continue

        config = TABLE_CONFIGS[table_format]
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
