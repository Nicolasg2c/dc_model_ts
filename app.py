from __future__ import annotations

import base64
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import run_etl_from_excel_bytes
from src.pipeline.config import DC_LABELS


def extract_key_hyperparameters(model) -> dict:
    """Extract key hyperparameters from a fitted sklearn pipeline model."""
    params = model.get_params()
    key_params = {}
    
    # Extract classifier parameters (prefixed with 'clf__')
    for key, value in params.items():
        if key.startswith('clf__'):
            # Clean up the key name
            clean_key = key.replace('clf__', '')
            key_params[clean_key] = value
    
    # Extract imputer strategy
    if 'imputer__strategy' in params:
        key_params['imputer_strategy'] = params['imputer__strategy']
    
    # Extract scaler info if present
    if 'scaler__with_mean' in params:
        key_params['scaler'] = 'StandardScaler'
    
    return key_params


def format_hyperparameters(params: dict) -> dict:
    """Format hyperparameters for display."""
    formatted = {}
    for key, value in params.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4g}"
        elif isinstance(value, str):
            formatted[key] = value
        else:
            formatted[key] = str(value)
    return formatted


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "SVM lineal": "modelo_svm_lineal_diagnostico.joblib",
    "SVM RBF": "modelo_svm_rbf_diagnostico.joblib",
    "Regresión logística": "modelo_logistic_regression_diagnostico.joblib",
    "Random forest": "modelo_random_forest_diagnostico.joblib",
}


st.set_page_config(
    page_title="Diagnóstico Deterioro cognitivo",
    page_icon="🧑🏻‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    [data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    button[aria-label*="sidebar" i],
    button[title*="sidebar" i],
    button[aria-label*="menu" i] {
        display: none;
    }
    .stApp {
        background: linear-gradient(180deg, #f6f8fb 0%, #eef2f6 100%);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #905ca4 0%, #7d20a1 55%, #7703a4 100%);
        color: #white;
        box-shadow: 0 18px 50px rgba(16, 42, 67, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .hero p {
        margin-top: 0.55rem;
        margin-bottom: 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    .card {
        background: white;
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08);
        border: 1px solid rgba(16, 42, 67, 0.06);
        margin-bottom: 1rem;
        color: #111827;
    }
    .section-title-dark {
        color: #111827;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .section-copy-dark {
        color: #111827;
        margin-bottom: 0.75rem;
    }
    .section-copy-dark p {
        color: #111827;
        margin: 0;
    }
    .top-right-logo {
        position: fixed;
        top: 0.8rem;
        right: 1rem;
        z-index: 9999;
        max-width: 80px;
    }
    .top-right-logo img {
        width: 100%;
        height: auto;
        display: block;
    }
    .disclaimer {
        margin-top: 1.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(17, 24, 39, 0.08);
        font-size: 0.85rem;
        color: #6b7280;
        text-align: center;
    }
    .stAlert {
        background-color: rgba(254, 202, 202, 0.8) !important;
        border: 1px solid rgba(185, 28, 28, 0.25) !important;
        color: #111827 !important;
    }
    .stAlert p,
    .stAlert div,
    .stAlert strong {
        color: #111827 !important;
    }
    .stMain [data-testid="stMetricLabel"],
    .stMain [data-testid="stMetricValue"],
    .stMain [data-testid="stMetricDelta"] {
        color: #111827 !important;
    }
    .stRadio,
    .stRadio label,
    .stRadio div,
    [data-testid="stRadio"],
    [data-testid="stRadio"] * {
        color: #929496 !important;
    }
    [data-testid="stSidebar"] {
        display: block !important;
        position: sticky;
        top: 0;
        height: 100vh;
        background: #000000;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] > div {
        background: #000000;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    [data-testid="stRadio"] {
        background: #ffffff;
        padding: 0.6rem 0.75rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(17, 24, 39, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def label_name(code: int | str) -> str:
    try:
        return DC_LABELS[int(code)]
    except Exception:
        return str(code)


@st.cache_resource(show_spinner=False)
def load_model(model_filename: str):
    return joblib.load(BASE_DIR / "models" / model_filename)


def parse_text_input(raw_text: str, expected_columns: list[str]) -> pd.DataFrame:
    text = raw_text.strip()
    if not text:
        raise ValueError("No se recibió texto para analizar.")

    candidates = [text]
    if ";" in text and "," not in text:
        candidates.append(text.replace(";", ","))

    for candidate in candidates:
        for separator in (",", ";", "\t"):
            try:
                frame = pd.read_csv(StringIO(candidate), sep=separator)
            except Exception:
                continue

            frame.columns = [str(column).strip() for column in frame.columns]
            if set(expected_columns).issubset(frame.columns):
                return frame

            if frame.shape[0] == 1 and frame.shape[1] == len(expected_columns):
                frame = pd.read_csv(StringIO(candidate), sep=separator, header=None)
                frame.columns = expected_columns
                return frame

    frame = pd.read_csv(StringIO(text), header=None)
    if frame.shape[1] != len(expected_columns):
        raise ValueError(
            "No se pudo interpretar el texto como una fila CSV compatible con el modelo."
        )
    frame.columns = expected_columns
    return frame


def build_sample_text(expected_columns: list[str]) -> str:
    sample_values = []
    for column in expected_columns:
        if column == "nivel_estudio":
            sample_values.append("1")
        elif column == "age_num":
            sample_values.append("74")
        else:
            sample_values.append("3")
    header = ",".join(expected_columns)
    values = ",".join(sample_values)
    return f"{header}\n{values}"


def align_input(frame: pd.DataFrame, expected_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    working = frame.copy()
    working.columns = [str(column).strip() for column in working.columns]
    extra_columns = [column for column in working.columns if column not in expected_columns]
    missing_columns = [column for column in expected_columns if column not in working.columns]
    aligned = working.reindex(columns=expected_columns)
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    return aligned, missing_columns, extra_columns


def extract_from_excel(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, df_mean, df_median = run_etl_from_excel_bytes(uploaded_file.getvalue(), verbose=False)
    return df_mean, df_median


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    threshold = 0.5

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        classes = list(model.classes_)
        for index, class_code in enumerate(classes):
            output[f"prob_{label_name(class_code)}"] = probabilities[:, index]

        best_class_indices = probabilities.argmax(axis=1)
        best_probabilities = probabilities.max(axis=1)
        raw_predictions = model.predict(frame)
        output["prediccion_modelo"] = raw_predictions
        output["prediccion_modelo_clinica"] = output["prediccion_modelo"].map(label_name)
        output["clase_max_prob"] = [classes[index] for index in best_class_indices]
        output["nombre_clase_max_prob"] = output["clase_max_prob"].map(label_name)
        output["probabilidad_maxima"] = best_probabilities
        output["umbral"] = threshold
        output["supera_umbral"] = best_probabilities >= threshold
        output["confianza"] = best_probabilities

    else:
        predictions = model.predict(frame)
        output = pd.DataFrame({"prediccion_codigo": predictions})
        output["prediccion_clinica"] = output["prediccion_codigo"].map(label_name)
        output["probabilidad_maxima"] = pd.NA
        output["umbral"] = threshold
        output["supera_umbral"] = pd.NA
        output["prediccion_modelo"] = predictions
        output["prediccion_modelo_clinica"] = output["prediccion_modelo"].map(label_name)

    return output


logo_path = BASE_DIR / "assets" / "images" / "UNAB LOGO.webp"
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

st.markdown(
    f"""
    <div class="top-right-logo">
        <img src="data:image/webp;base64,{logo_b64}" />
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>APLICACIÓN DE MODELOS DE MACHINE LEARNING PARA LA DETECCIÓN Y CLASIFICACIÓN DEL DETERIORO COGNITIVO MEDIANTE EL USO EVALUACIONES NEUROPSICOLÓGICAS EN ADULTOS MAYORES EN COLOMBIA</h1>
        <p> Esta aplicación puede ser utilizada como herramienta de apoyo para la detección y clasificación del deterioro cognitivo en adultos mayores, utilizando modelos de machine learning entrenados con evaluaciones neuropsicológicas. Los resultados obtenidos a través de esta aplicación deben ser interpretados por profesionales de la salud y no sustituyen la evaluación clínica directa.</p>
        <p>Aviso Médico: Esta información describe el comportamiento interno de librerías de programación y optimización matemática. No representa un diagnóstico ni debe aplicarse para validar la condición clínica real de un sujeto.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


selected_model_name = st.sidebar.selectbox(
    "Modelo",
    list(MODEL_FILES.keys()),
    index=0,
)
model = load_model(MODEL_FILES[selected_model_name])
expected_columns = [str(column) for column in getattr(model, "feature_names_in_", [])]

st.sidebar.markdown("### Columnas")
st.sidebar.caption(
    "Los modelos utilizan las siguientes variables en este orden. Las columnas faltantes se imputaran automáticamente."
)
st.sidebar.write(expected_columns)

# Display model hyperparameters in sidebar
st.sidebar.markdown("### Hiperparámetros del modelo")
st.sidebar.caption("Configuración principal del modelo seleccionado.")
hyperparams = extract_key_hyperparameters(model)
formatted_params = format_hyperparameters(hyperparams)

# Display as a table using DataFrame
params_df = pd.DataFrame(list(formatted_params.items()), columns=["Parámetro", "Valor"])
st.sidebar.dataframe(params_df, width="stretch", hide_index=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
left, right = st.columns([1.2,0.8])

with left:
    st.markdown('<div class="section-title-dark">Cargue de datos</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Modo de carga",
        ["Subir Excel y Extraer", "Pegar texto CSV", "Subir archivo CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # st.markdown('<div class="section-copy-dark">Modo de carga</div>', unsafe_allow_html=True)

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = build_sample_text(expected_columns)

    source_frame = None
    extracted_message = None

    if input_mode == "Subir Excel y Extraer":
        uploaded_excel = st.file_uploader("", type=["xlsx", "xls"])
        st.markdown('<div class="section-copy-dark">Sube un archivo Excel</div>', unsafe_allow_html=True)
        if uploaded_excel is not None:
            try:
                source_frame, df_median = extract_from_excel(uploaded_excel)
                extracted_message = (
                    f"Se extrajeron {len(source_frame)} registros con el pipeline de preprocesamiento de datos. "
                    f"Se usará el conjunto de datos que usa la media para la reconstrucción de los dominios cognitivos para inferencia y el conjunto de datos que usa la mediana también quedó disponible en memoria."
                )
            except Exception as exc:
                st.error(f"No se pudo procesar el Excel: {exc}")
    elif input_mode == "Pegar texto CSV":
        if st.button("Cargar ejemplo en el texto"):
            st.session_state["input_text"] = build_sample_text(expected_columns)

        st.markdown(
            "<div class='section-copy-dark'>Pega una tabla CSV con encabezados o una sola fila en el orden de las columnas esperadas</div>",
            unsafe_allow_html=True,
        )
        raw_text = st.text_area(
            "",
            key="input_text",
            height=220,
            label_visibility="collapsed",
        )
        if raw_text.strip():
            try:
                source_frame = parse_text_input(raw_text, expected_columns)
            except Exception as exc:
                st.error(str(exc))
    else:
        uploaded_file = st.file_uploader("", type=["csv"])
        st.markdown('<div class="section-copy-dark">Sube un archivo CSV</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                source_frame = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")

with right:
    st.markdown('<div class="section-title-dark">Salida del modelo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy-dark">Se muestran las probabilidades de cada clase en base a la entrada y al modelo usado para la inferencia.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

if extracted_message:
    st.markdown(f'<div class="section-copy-dark">{extracted_message}</div>', unsafe_allow_html=True)

if source_frame is not None:
    if input_mode == "Subir Excel y Extraer":
        aligned_frame = source_frame.reindex(columns=expected_columns)
        aligned_frame = aligned_frame.apply(pd.to_numeric, errors="coerce")
        missing_columns = [column for column in expected_columns if column not in source_frame.columns]
        extra_columns = [column for column in source_frame.columns if column not in expected_columns]
    else:
        aligned_frame, missing_columns, extra_columns = align_input(source_frame, expected_columns)

    if extra_columns:
        st.warning(f"Se ignoraron columnas no usadas por el modelo: {', '.join(extra_columns)}")
    if missing_columns:
        st.info(f"Se completaron con valores faltantes estas columnas: {', '.join(missing_columns)}")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f'<div class="section-copy-dark">Vista previa de la entrada alineada al modelo</div>', unsafe_allow_html=True)
    st.dataframe(aligned_frame, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        predictions = predict_frame(model, aligned_frame)
    except Exception as exc:
        st.error(f"Error durante la inferencia: {exc}")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(predictions) == 1:
            top_row = predictions.iloc[0]
            metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
            metric_col_1.metric("Predicción del modelo", top_row["prediccion_modelo_clinica"])
            metric_col_2.metric("Clase de mayor probabilidad", top_row["nombre_clase_max_prob"])
            metric_col_3.metric("Prob. máxima", f"{top_row['probabilidad_maxima']:.2%}" if pd.notna(top_row.get("probabilidad_maxima")) else "N/D")
            metric_col_4.metric("¿Supera umbral?", "Sí" if bool(top_row.get("supera_umbral")) else "No" if pd.notna(top_row.get("supera_umbral")) else "N/D")
            st.caption("La predicción del modelo y la clase de mayor probabilidad pueden diferir según el clasificador y la separación de clases.")
        st.dataframe(predictions, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultados CSV",
            data=csv_bytes,
            file_name=f"predicciones_{selected_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        probability_columns = [column for column in predictions.columns if column.startswith("prob_")]
        if probability_columns:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            selected_index = 0 if len(predictions) == 1 else st.selectbox(
                "Selecciona una fila para ver probabilidades",
                list(range(len(predictions))),
            )
            probability_view = predictions.loc[[selected_index], probability_columns].T
            probability_view.columns = ["probabilidad"]
            st.bar_chart(probability_view, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer">
        Copyright © 2026 UNAB. All Right Reserved By Nicolás Gutierrez.
    </div>
    """,
    unsafe_allow_html=True,
)
