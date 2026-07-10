"""
cleaning.py — Limpieza y estandarización de hojas Excel.

Paso 2 del pipeline ETL:
  - Elimina filas completamente vacías de cada hoja.
  - Detecta el formato de tabla (Tabla 0 / Tabla 1) por hoja.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .utils import clean_value


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
