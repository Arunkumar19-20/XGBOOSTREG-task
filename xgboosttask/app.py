import os
import joblib
import pandas as pd
from flask import Flask, render_template, request


# --------------------------------
# App Setup
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# --------------------------------
# Load Model
# --------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# --------------------------------
# Home Route
# --------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        try:
            # Get values from form
            lotarea = float(request.form["lotarea"])
            bedrooms = int(request.form["bedrooms"])
            fullbath = int(request.form["fullbath"])
            yearbuilt = int(request.form["yearbuilt"])
            basement = float(request.form["basement"])
            overallqual = int(request.form["overallqual"])
            grlivarea = float(request.form["grlivarea"])
            garagecars = int(request.form["garagecars"])

            # Create input in training order
            input_data = pd.DataFrame(
                [[
                    grlivarea,     # GrLivArea
                    bedrooms,      # BedroomAbvGr
                    fullbath,      # FullBath
                    basement,      # TotalBsmtSF
                    garagecars,    # GarageCars
                    yearbuilt,     # YearBuilt
                    lotarea,       # LotArea
                    overallqual    # OverallQual
                ]]
            )

            # Predict (convert to numpy)
            pred = model.predict(input_data.values)[0]

            prediction = f"₹ {pred:,.0f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


# --------------------------------
# Run App
# --------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    app.run(host="0.0.0.0", port=port, debug=True)
