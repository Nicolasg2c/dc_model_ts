"""
imputation.py — Imputación clínica de valores nulos.

Paso 6 del pipeline ETL:
  - Genera columnas indicadoras `{col}_missing` para toda columna con nulos.
  - Columnas con tasa de nulos ≥ `high_missing` NO se imputan.
  - Categóricas  → moda por grupo clínico; fallback: moda global.
  - Numéricas    → mediana por grupo clínico; fallback: mediana global.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd



#Imputación clínica
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

# Imputación clínica de valores nulos
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
