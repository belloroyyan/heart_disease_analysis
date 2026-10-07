from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# --------------------------------------------------
# Load trained model and preprocessor
# --------------------------------------------------
# The notebook saved only the KNeighborsClassifier. It was trained on data
# that had first been passed through the fitted ColumnTransformer `ct`
# (StandardScaler on the numeric columns + OneHotEncoder on the categorical
# ones), so new input must go through that same fitted `ct` before the model.

try:
    preprocessor = joblib.load("model/preprocessor.joblib")
    model = joblib.load("model/heart_disease_model.joblib")
except FileNotFoundError as error:
    raise SystemExit(
        f"{error}\nPut heart_disease_model.joblib and preprocessor.joblib "
        "in the model/ folder. In the notebook, save the fitted preprocessor with:\n"
        "    joblib.dump(ct, root / 'model' / 'preprocessor.joblib')"
    )

# Catch a mismatched pair of files early instead of failing on a prediction.
n_expected = len(preprocessor.get_feature_names_out())
if model.n_features_in_ != n_expected:
    raise SystemExit(
        f"Model expects {model.n_features_in_} features but the preprocessor "
        f"produces {n_expected}. They were not saved from the same run."
    )

# In Heart_disease_statlog.csv, target 1 = heart disease present.
POSITIVE_CLASS = 1


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Receive values from the HTML form (the 13 UCI Heart Disease features).
    try:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])
    except (KeyError, ValueError):
        return render_template(
            "index.html",
            error="Please fill in every field with a valid number.",
        ), 400

    # Never trust the browser: check the values are in the allowed ranges.
    checks = [
        1 <= age <= 120,
        sex in (0, 1),
        cp in (0, 1, 2, 3),
        50 <= trestbps <= 300,
        50 <= chol <= 700,
        fbs in (0, 1),
        restecg in (0, 1, 2),
        40 <= thalach <= 250,
        exang in (0, 1),
        0 <= oldpeak <= 10,
        slope in (0, 1, 2),
        ca in (0, 1, 2, 3),
        thal in (1, 2, 3),
    ]

    if not all(checks):
        return render_template(
            "index.html",
            error="One or more values are outside the allowed range.",
        ), 400

    # Raw feature DataFrame with the same column names used in the notebook.
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    # Same two steps as the notebook: transform, then classify.
    transformed = preprocessor.transform(input_data)

    # Probability assigned to the "heart disease" class.
    probabilities = model.predict_proba(transformed)[0]
    classes = list(model.classes_)
    disease_probability = probabilities[classes.index(POSITIVE_CLASS)]

    prediction = int(disease_probability >= 0.5)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(disease_probability * 100, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
