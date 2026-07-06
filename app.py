import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "kmeans_riesgo_actuarial.pkl"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
CLUSTERS_CSV_PATH = BASE_DIR / "outputs" / "insurance_con_clusters.csv"

REGIONES = ["southeast", "southwest", "northeast", "northwest"]

EXPLICACIONES = {
    "Bajo": "Este cliente fue agrupado con perfiles de menor costo médico promedio.",
    "Medio": "Este cliente fue agrupado con perfiles de costo y factores de riesgo intermedios.",
    "Alto": "Este cliente fue agrupado con perfiles de mayor costo médico promedio y/o factores de riesgo relevantes.",
}

COLOR_RIESGO = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}


@st.cache_resource
def load_model():
    try:
        modelo = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # las llaves del mapa de riesgo se guardaron como string en el JSON
        mapa_riesgo = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
        return modelo, metadata, mapa_riesgo
    except FileNotFoundError as e:
        st.error(
            "No se encontraron los archivos del modelo. Verificá que la carpeta "
            "'models' (con kmeans_riesgo_actuarial.pkl y model_metadata.json) "
            f"esté junto a app.py.\n\nDetalle: {e}"
        )
        st.stop()


@st.cache_data
def load_clusters_csv():
    try:
        return pd.read_csv(CLUSTERS_CSV_PATH)
    except FileNotFoundError:
        st.warning(
            "No se encontró 'outputs/insurance_con_clusters.csv', así que no se "
            "podrá mostrar la comparación con el resto de clientes."
        )
        return None


def evaluar_cliente(modelo, mapa_riesgo, age, sex, bmi, children, smoker, region, charges):
    cliente = pd.DataFrame([{
        "age": age,
        "sex": str(sex).lower(),
        "bmi": bmi,
        "children": children,
        "smoker": str(smoker).lower(),
        "region": str(region).lower(),
        "charges": charges,
    }])

    cluster = int(modelo.predict(cliente)[0])
    riesgo = mapa_riesgo[cluster]

    return cluster, riesgo, EXPLICACIONES[riesgo]


st.set_page_config(page_title="Riesgo Actuarial - Clustering", page_icon="🏥")

st.title("Clasificador de Riesgo Actuarial")
st.caption("Arleth Adyani Chevez Bonilla — Cuenta: 20221900251")

st.write(
    "Esta aplicación agrupa el perfil de un cliente de seguro médico según su "
    "nivel de riesgo actuarial (**Bajo**, **Medio** o **Alto**), usando un modelo "
    "de clustering (K-means) entrenado sobre el dataset `insurance.csv`."
)

modelo, metadata, mapa_riesgo = load_model()
df_clusters = load_clusters_csv()

with st.expander("ℹ️ Sobre el modelo"):
    st.write(f"**Tipo de modelo:** {metadata['tipo_modelo']}")
    st.write(f"**Número de clusters:** {metadata['n_clusters']}")
    st.write(f"**Silhouette score:** {metadata['silhouette_score']}")
    st.write(
        "El nivel de riesgo no viene etiquetado en los datos originales: se "
        "asignó según el promedio de cargos médicos de cada cluster (a mayor "
        "cargo promedio, mayor riesgo)."
    )

st.subheader("Datos del cliente")

with st.form("form_cliente"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Sexo", ["male", "female"])
        bmi = st.number_input("BMI (índice de masa corporal)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Número de hijos", min_value=0, max_value=10, value=0, step=1)

    with col2:
        smoker = st.selectbox("¿Fumador?", ["yes", "no"])
        region = st.selectbox("Región", REGIONES)
        charges = st.number_input("Cargos médicos (charges, en USD)", min_value=0.0, value=10000.0, step=100.0)

    enviado = st.form_submit_button("Evaluar riesgo")

if enviado:
    cluster, riesgo, explicacion = evaluar_cliente(
        modelo, mapa_riesgo, age, sex, bmi, children, smoker, region, charges
    )

    st.subheader("Resultado")
    st.success(f"{COLOR_RIESGO[riesgo]} **Riesgo actuarial: {riesgo}**  (Cluster {cluster})")
    st.write(explicacion)

    if df_clusters is not None:
        st.subheader("Comparación con la cartera de clientes")

        resumen = df_clusters.groupby("riesgo_actuarial").agg(
            cantidad_clientes=("riesgo_actuarial", "count"),
            cargos_promedio=("charges", "mean"),
            edad_promedio=("age", "mean"),
            bmi_promedio=("bmi", "mean"),
        ).round(2)

        st.write("Este cliente se compara así frente al resto de la cartera:")
        st.dataframe(resumen)

        st.write("Distribución de cargos médicos promedio por nivel de riesgo:")
        st.bar_chart(resumen["cargos_promedio"])

        st.caption(
            f"El cliente evaluado tiene charges = {charges:,.2f}, frente a un "
            f"promedio de {resumen.loc[riesgo, 'cargos_promedio']:,.2f} en su "
            f"grupo de riesgo ({riesgo})."
        )
