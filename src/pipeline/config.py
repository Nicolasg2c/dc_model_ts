"""
config.py — Configuración, constantes y mapeos del dominio neuropsicológico.
"""

from typing import Dict, List

# Constantes de dominio — columnas de identificación de cada paciente

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

# Etiquetas y paleta para visualizaciones
DC_LABELS: Dict[int, str] = {0: "Control", 1: "DCL", 2: "Demencia"}
DC_ORDER: List[int] = [0, 1, 2]
EDUCATION_LEVEL: Dict[int, str] = {
    0: "Escolaridad Baja",
    1: "Escolaridad Alta",
}

# Configuración de columnas según formato de tabla
TABLE_CONFIGS: Dict[int, Dict[str, int]] = {
    0: {"headers_col": 0, "value_col": 3, "nivel_estudio": 0},
    1: {"headers_col": 1, "value_col": 4, "nivel_estudio": 1},
}

# Mapas de valores de normalización de las tablas 0 y 1
REEMPLAZOS_TABLA_1: Dict[str, str] = {
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

REEMPLAZOS_TABLA_0: Dict[str, str] = {
    "alto": "alto",
    "deficit": "alteracion_severa",
    "promedio": "promedio",
    "bajo": "bajo",
    "maximo": "alto",
}

# Mapa ordinal para las 4 categorías clínicas
ORDINAL_MAP: Dict[str, int] = {
    "alteracion_severa": 1,
    "bajo": 2,
    "promedio": 3,
    "alto": 4,
}
