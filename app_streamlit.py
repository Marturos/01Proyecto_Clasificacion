import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path

# =========================
# App Streamlit - ENAHO 2024
# Proyecto de clasificación
# =========================

st.set_page_config(
    page_title="Clasificación ENAHO 2024 - NBI",
    page_icon="🏠",
    layout="wide"
)


# Bloque: cargar el modelo entrenado desde la misma carpeta de app_streamlit.py
@st.cache_resource
def cargar_modelo():
    model_path = Path(__file__).resolve().parent / "modelo_enaho_nbi.joblib"

    if not model_path.exists():
        st.error(f"No se encontró el modelo en la ruta: {model_path}")
        st.stop()

    paquete = load(model_path)

    if "model" not in paquete or "metadata" not in paquete:
        st.error("El archivo modelo_enaho_nbi.joblib no contiene las claves 'model' y 'metadata'.")
        st.stop()

    return paquete["model"], paquete["metadata"]


# Bloque: cargar modelo y metadata para usar la app
clf, metadata = cargar_modelo()

st.title("Modelo de Clasificación ENAHO 2024")
st.markdown(
    """
    Esta aplicación predice si un hogar presenta al menos una Necesidad Básica Insatisfecha (NBI).

    **Target:** `TARGET_NBI`  
    **0:** No presenta NBI  
    **1:** Presenta al menos una NBI
    """
)
st.markdown("---")

features = metadata["features"]
cat_features = metadata["categorical_features"]
num_features = metadata["numeric_features"]
cat_options = metadata["categorical_options"]
num_ranges = metadata["numeric_ranges"]
labels = metadata.get("feature_labels", {})

st.sidebar.header("Formulario de predicción")

sample_cases = metadata.get("sample_cases", [])
sample_names = ["Sin caso de prueba"] + [f"Caso {i+1}" for i in range(len(sample_cases))]
selected_sample = st.sidebar.selectbox(
    "Cargar caso de prueba para comparar con VS Code",
    sample_names,
)

default_values = {}
if selected_sample != "Sin caso de prueba":
    idx = sample_names.index(selected_sample) - 1
    default_values = sample_cases[idx]

st.sidebar.markdown("### Variables categóricas")
input_data = {}

for col in cat_features:
    options = [str(x) for x in cat_options[col]]
    default = str(default_values.get(col, options[0]))
    index = options.index(default) if default in options else 0
    input_data[col] = st.sidebar.selectbox(
        f"{labels.get(col, col)}",
        options=options,
        index=index,
    )

st.sidebar.markdown("### Variables numéricas")

for col in num_features:
    r = num_ranges[col]
    min_v = float(r["min"])
    max_v = float(r["max"])
    median_v = float(r["median"])
    default = float(default_values.get(col, median_v))

    if default < min_v:
        default = min_v

    if default > max_v:
        default = max_v

    input_data[col] = st.sidebar.number_input(
        f"{labels.get(col, col)}",
        min_value=min_v,
        max_value=max_v,
        value=default,
        step=1.0,
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Datos ingresados")
    obs = pd.DataFrame([input_data], columns=features)
    st.dataframe(obs, use_container_width=True)

    predecir = st.button("Predecir", type="primary")

with col2:
    st.subheader("Información del modelo")
    st.write(f"**Modelo seleccionado:** {metadata['best_model']}")
    st.write(f"**Base usada:** {metadata['dataset']}")
    st.write(f"**Filas usadas para modelamiento:** {metadata.get('n_rows_modeling', 'No disponible')}")
    st.write(f"**Definición del target:** {metadata['target_definition']}")

if predecir:
    pred = int(clf.predict(obs)[0])
    proba_1 = float(clf.predict_proba(obs)[0][1])
    proba_0 = float(clf.predict_proba(obs)[0][0])

    st.markdown("---")
    st.subheader("Resultado de la predicción")

    if pred == 1:
        st.error(f"Clase predicha: 1 - Hogar con al menos una NBI. Probabilidad: {proba_1:.4f}")
    else:
        st.success(f"Clase predicha: 0 - Hogar sin NBI. Probabilidad: {proba_0:.4f}")

    st.write("Probabilidades por clase:")
    st.dataframe(
        pd.DataFrame({
            "Clase": ["0 - Sin NBI", "1 - Con NBI"],
            "Probabilidad": [proba_0, proba_1],
        }),
        use_container_width=True,
    )

    if selected_sample != "Sin caso de prueba":
        real = default_values.get("TARGET_REAL", "No disponible")
        pred_vs = default_values.get("PREDICCION_MODELO", "No disponible")
        prob_vs = default_values.get("PROBABILIDAD_TARGET_1", "No disponible")
        st.info(
            f"Comparación con VS Code para {selected_sample}: "
            f"target real={real}, predicción guardada={pred_vs}, "
            f"probabilidad clase 1={prob_vs}."
        )

st.markdown("---")
st.subheader("Importancia de variables")

importance = pd.DataFrame(metadata["feature_importance"])

if not importance.empty:
    importance_plot = importance.set_index("feature")["importance"].sort_values(ascending=True)
    st.bar_chart(importance_plot)
    st.dataframe(importance, use_container_width=True)
else:
    st.info("No hay importancia de variables disponible para mostrar.")

st.markdown("---")

with st.expander("Auditoría técnica del despliegue"):
    st.markdown(
        """
        - La aplicación carga un único archivo `modelo_enaho_nbi.joblib` que contiene el pipeline completo.
        - El pipeline incluye preprocesamiento de variables numéricas y categóricas.
        - Las variables categóricas se procesan con OneHotEncoder.
        - Las variables numéricas se imputan y escalan.
        - El modelo final fue seleccionado comparando Random Forest y Gradient Boosting.
        - El formulario usa `st.sidebar` con listas desplegables y entradas numéricas.
        - Los casos de prueba permiten comprobar que la predicción web coincide con VS Code.
        """
    )
