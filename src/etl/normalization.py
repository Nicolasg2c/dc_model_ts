"""
normalization.py — Normalización de valores categóricos clínicos.

Paso 5 del pipeline ETL:
  - Aplica limpieza de texto a las columnas de features.
  - Mapea los valores crudos a las 4 categorías canónicas:
    alteracion_severa | bajo | promedio | alto.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import ID_COLS, REEMPLAZOS_TABLA_0, REEMPLAZOS_TABLA_1
from .utils import limpiar_texto


# ---------------------------------------------------------------------------
# 5. Normalización — estandarización de valores categóricos
# ---------------------------------------------------------------------------


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


def normalize_tabla_0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los valores categóricos de la Tabla 0 (escolaridad baja).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame crudo de Tabla 0.

    Retorna
    -------
    pd.DataFrame
        DataFrame con valores normalizados a las 4 categorías canónicas.
    """
    return _normalizar_df(df, REEMPLAZOS_TABLA_0)


def normalize_tabla_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los valores categóricos de la Tabla 1 (escolaridad alta).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame crudo de Tabla 1.

    Retorna
    -------
    pd.DataFrame
        DataFrame con valores normalizados a las 4 categorías canónicas.
    """
    return _normalizar_df(df, REEMPLAZOS_TABLA_1)
