"""
services/__init__.py — API pública del paquete de servicios ETL.

Uso rápido
----------
    from src.services import run_etl

    df_t0, df_t1, df_complete = run_etl(dotenv_path=".env")
"""

from .config import (
    DC_LABELS,
    DC_ORDER,
    DOMINIOS,
    EDUCATION_LEVEL,
    FEATURES_TABLA_0,
    FEATURES_TABLA_1,
    ID_COLS,
    ORDINAL_MAP,
    REEMPLAZOS_TABLA_0,
    REEMPLAZOS_TABLA_1,
    TABLE_CONFIGS,
)
from .cleaning import clean_sheets, detect_table_format
from .features import extract_features_from_table, extract_patient_info, search_values
from .imputation import imputacion_null, null_data_info
from .normalization import normalize_tabla_0, normalize_tabla_1
from .utils import clean_value, limpiar_texto
from .etl import build_df_complete, load_excel_from_github, run_etl

__all__ = [
    # config
    "ID_COLS",
    "FEATURES_TABLA_0",
    "FEATURES_TABLA_1",
    "DOMINIOS",
    "DC_LABELS",
    "DC_ORDER",
    "EDUCATION_LEVEL",
    "TABLE_CONFIGS",
    "REEMPLAZOS_TABLA_0",
    "REEMPLAZOS_TABLA_1",
    "ORDINAL_MAP",
    # utils
    "clean_value",
    "limpiar_texto",
    # cleaning
    "clean_sheets",
    "detect_table_format",
    # features
    "extract_patient_info",
    "extract_features_from_table",
    "search_values",
    # normalization
    "normalize_tabla_0",
    "normalize_tabla_1",
    # imputation
    "null_data_info",
    "imputacion_null",
    # etl (orchestrator)
    "load_excel_from_github",
    "build_df_complete",
    "run_etl",
]
