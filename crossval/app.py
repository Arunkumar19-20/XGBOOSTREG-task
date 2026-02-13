import streamlit as st
import numpy as np
import joblib

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# =====================================
# Custom CSS (Wine Theme)
# =====================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #3a0d0d, #7b1e1e);
}

h1 {
    color: white;
    text-align: center;
}

.result-card {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =====================================
# Title
# =====================================
st.title("🍷 Wine Quality Prediction App")
st.markdown("<br>", unsafe_allow_html=True)

# =====================================
# Load Model
# =====================================
model = joblib.load("wine_quality.pkl")

# =====================================
# Layout
# =====================================
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
    density = st.number_input("Density", 0.9900, 1.0100, 0.9960, format="%.4f")
    pH = st.number_input("pH", 2.0, 4.5, 3.3)
    sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.6)
    alcohol = st.number_input("Alcohol (%)", 5.0, 15.0, 10.0)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================
# Prediction
# =====================================
if st.button("🔮 Predict Wine Quality"):

    input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    prediction = model.predict(input_data)[0]

    # Quality Interpretation
    if prediction <= 4:
        quality_label = "Low Quality"
        color = "red"
    elif prediction <= 6:
        quality_label = "Medium Quality"
        color = "orange"
    else:
        quality_label = "High Quality"
        color = "green"

    st.markdown(f"""
        <div class="result-card">
            <h2>Predicted Wine Quality</h2>
            <h1 style="color:#7b1e1e;">{prediction:.2f}</h1>
            <h2 style="color:{color};">{quality_label}</h2>
        </div>
    """, unsafe_allow_html=True)
