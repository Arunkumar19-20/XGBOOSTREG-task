import streamlit as st
import numpy as np
import joblib

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Wine DBSCAN Clustering",
    page_icon="🍷",
    layout="wide"
)

# =====================================
# Custom CSS (Premium Gradient UI)
# =====================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #1e3c72, #2a5298);
}
h1 {
    color: white;
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stNumberInput label {
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.success {
    background-color: #28a745;
    color: white;
}
.error {
    background-color: #dc3545;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍷 Wine Quality Clustering (DBSCAN)</h1>", unsafe_allow_html=True)
st.write("### Enter Wine Chemical Properties")

# =====================================
# Load Model
# =====================================
db_model = joblib.load("db_model.pkl")

# =====================================
# Input Section (Card Style Layout)
# =====================================
col1, col2, col3 = st.columns(3)

with col1:
    alcohol = st.number_input("Alcohol", value=13.0)
    malic_acid = st.number_input("Malic Acid", value=2.0)
    ash = st.number_input("Ash", value=2.3)
    ash_alcanity = st.number_input("Ash Alcanity", value=19.0)
    magnesium = st.number_input("Magnesium", value=100.0)

with col2:
    total_phenols = st.number_input("Total Phenols", value=2.5)
    flavanoids = st.number_input("Flavanoids", value=2.0)
    nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.3)
    proanthocyanins = st.number_input("Proanthocyanins", value=1.5)

with col3:
    color_intensity = st.number_input("Color Intensity", value=5.0)
    hue = st.number_input("Hue", value=1.0)
    od280 = st.number_input("OD280/OD315", value=3.0)
    proline = st.number_input("Proline", value=800.0)

# =====================================
# Prediction Button
# =====================================
if st.button("🚀 Predict Cluster"):

    input_data = np.array([[alcohol, malic_acid, ash, ash_alcanity,
                            magnesium, total_phenols, flavanoids,
                            nonflavanoid_phenols, proanthocyanins,
                            color_intensity, hue, od280, proline]])

    try:
        cluster = db_model.fit_predict(input_data)

        # Professional cluster naming
        cluster_details = {
            0: ("🍷 Classic Reserve", "Balanced structure with refined chemical composition."),
            1: ("🍇 Premium Vintage", "Rich intensity and elevated phenolic content."),
            2: ("🌿 Elegant Blend", "Smooth profile with moderate characteristics."),
            -1: ("✨ Rare Signature", "Distinct chemical pattern outside major wine groups.")
        }

        name, description = cluster_details.get(
            cluster[0],
            ("🍾 Special Selection", "Unique wine profile identified.")
        )

        # Display Result
        if cluster[0] == -1:
            st.markdown(
                f"""
                <div class="result-box error">
                    {name}<br>
                    <span style="font-size:16px;">{description}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box success">
                    {name}<br>
                    <span style="font-size:16px;">{description}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error("Model prediction could not be completed.")
