import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Milk Quality Prediction",
    page_icon="🥛",
    layout="centered"
)


# ---------------------------------
# Load Dataset
# ---------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/arun/Desktop/XGBOOSTREG-task/xgboostclass/milk_quality_data.csv")
    return df


df = load_data()


# ---------------------------------
# Prepare Data
# ---------------------------------

X = df[['ph', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'colour']]
y = df['grade']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------
# Load Trained Model
# ---------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("milk_quality_model.pkl")   # Make sure file is in same folder
    return model


model = load_model()


# ---------------------------------
# Accuracy (Using Loaded Model)
# ---------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------------
# UI Styling
# ---------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    /* Main container */
    .block-container {
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }

    /* Headings */
    h1, h2, h3 {
        color: #ffffff;
        text-align: center;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(to right, #ff512f, #dd2476);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        opacity: 0.9;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4e54c8, #8f94fb);
        color: white;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffd369;
        font-size: 28px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ---------------------------------
# Title
# ---------------------------------

st.title("🥛 Milk Quality Prediction System")
st.write("AI-based Milk Grade Classification")


# ---------------------------------
# Sidebar Inputs
# ---------------------------------

st.sidebar.header("📊 Enter Milk Parameters")


ph = st.sidebar.slider("pH", 0.0, 14.0, 6.5)
temperature = st.sidebar.slider("Temperature (°C)", 0, 100, 25)
taste = st.sidebar.slider("Taste (1-10)", 1, 10, 5)
odor = st.sidebar.slider("Odor (1-10)", 1, 10, 5)
fat = st.sidebar.slider("Fat (1-10)", 1, 10, 5)
turbidity = st.sidebar.slider("Turbidity (1-10)", 1, 10, 5)
colour = st.sidebar.slider("Colour (1-10)", 1, 10, 5)


# ---------------------------------
# Input Data
# ---------------------------------

input_data = pd.DataFrame({
    "ph": [ph],
    "temperature": [temperature],
    "taste": [taste],
    "odor": [odor],
    "fat": [fat],
    "turbidity": [turbidity],
    "colour": [colour]
})


# ---------------------------------
# Predict
# ---------------------------------

if st.button("🔮 Predict Quality"):

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)

    confidence = proba.max() * 100

    grade_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    st.success("✅ Prediction Successful")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📌 Grade", grade_map[pred])

    with col2:
        st.metric("🎯 Confidence", f"{confidence:.2f} %")


# ---------------------------------
# Model Performance
# ---------------------------------

st.markdown("---")

st.subheader("📈 Model Accuracy")

st.write(f"**Accuracy on Test Data:** {accuracy:.4f}")

st.progress(int(accuracy * 100))


# ---------------------------------
# Dataset Preview
# ---------------------------------

with st.expander("📂 Dataset Preview"):
    st.dataframe(df.head(20))
