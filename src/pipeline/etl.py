"""
etl.py — Pipeline ETL para el procesamiento de datos neuropsicológicos.

Pasos principales
-----------------
1. Extracción: Descarga el archivo Excel desde GitHub.
2. Limpieza: Elimina filas vacías y detecta el formato de cada hoja.
3. Extracción: Obtiene los features neuropsicológicos de cada hoja.
4. Normalización: Estandariza los valores categóricos (interpretación clínica).
5. Imputación: Imputa nulos con mediana/moda por grupo clínico.
6. Integración: Construye `df_complete` con dominios cognitivos agregados.

Retorno de `run_etl()`
----------------------
Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (df_tabla_0_imp, df_tabla_1_imp, df_complete)
"""

from __future__ import annotations

import warnings
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import requests
from dotenv import dotenv_values

from .cleaning import clean_sheets, detect_table_format
from .config import (
    DOMINIOS,
    FEATURES_TABLA_0,
    FEATURES_TABLA_1,
    ID_COLS,
    ORDINAL_MAP,
)
from .features import search_values
from .imputation import imputacion_null, null_data_info
from .normalization import normalize_tabla_0, normalize_tabla_1

warnings.filterwarnings("ignore")



#Extracción de datos desde GitHub
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


#Construcción de df_complete con dominios cognitivos


def _to_ordinal(series: pd.Series) -> pd.Series:
    """Convierte una columna categórica al mapa ordinal de 4 niveles."""
    return series.map(ORDINAL_MAP)


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

    # Eliminar columna age
    df_complete.drop(columns=["age"], inplace=True)

    return df_complete



# Función principal etl
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
    # 1. Extracción
    if verbose:
        print("Descargando datos desde GitHub...")
    xlsx_data = load_excel_from_github(dotenv_path)

    # 2. Limpieza de hojas
    cleaned_sheets = clean_sheets(xlsx_data)
    if verbose:
        print(f"Hojas cargadas: {len(cleaned_sheets)}")


    # 3. Detección de formato de cada tabla
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

    # 4. Extracción de features por tabla
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

    # 5. Normalización de valores categóricos
    df_tabla_0 = normalize_tabla_0(df_tabla_0)
    df_tabla_1 = normalize_tabla_1(df_tabla_1)

    # 6. Imputación clínica
    if verbose:
        print("\n Imputando nulos...")
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

    # 7. Construcción del dataframe conjunto
    df_complete = build_df_complete(df_tabla_0_imp, df_tabla_1_imp)

    if verbose:
        print(f"\ndf_complete listo: {df_complete.shape}")
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


#función principal para ejecutar el pipeline ETL
if __name__ == "__main__":
    df_tabla_0_imp, df_tabla_1_imp, df_complete = run_etl(verbose=True)
    print("\nPrimeras filas de df_complete:")
    print(df_complete.head())
