"""
utils.py — Funciones utilitarias de normalización de texto.

Contiene helpers de uso general para limpiar y estandarizar cadenas de
caracteres, utilizados transversalmente por los demás módulos del pipeline.
"""

from __future__ import annotations

import unicodedata
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers de normalización de texto
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
