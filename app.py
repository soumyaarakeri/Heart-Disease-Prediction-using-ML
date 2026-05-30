from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import json

# FIX: Disable GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create the images folder
IMAGE_DIR = "static/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# -----------------------
# In-memory user storage
# -----------------------
users = {}

# -----------------------
# Load trained model
# -----------------------
model = pickle.load(open("model/trained_model.pkl", "rb"))

# Load model metrics
with open("model/metrics.json", "r") as f:
    metrics = json.load(f)


# -----------------------
# ROUTES
# -----------------------

@app.route("/", methods=["GET"])
def signup():
    return render_template("signup.html")


@app.route("/signup", methods=["POST"])
def signup_post():
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")

    if password != confirm_password:
        return "Passwords do not match!"

    if username in users:
        return "Username already exists!"

    users[username] = {"email": email, "password": password}
    return redirect(url_for("login"))


@app.route("/login.html", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/login.html", methods=["POST"])
def login_post():
    username = request.form.get("username")
    password = request.form.get("password")

    user = users.get(username)
    if user and user["password"] == password:
        session["username"] = username
        return redirect(url_for("index"))
    return "Invalid credentials!"


@app.route("/index.html")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/project_info")
def project_info():
    return render_template("project_info.html")


@app.route("/about_heart")
def about_heart():
    return render_template("about_heart.html")


@app.route("/prevention_tips")
def prevention_tips():
    return render_template("prevention_tips.html")


# -----------------------
# PREDICT
# -----------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        # Get user input
        data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        input_data = np.array([data])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "⚠ High chance of Heart Disease" if prediction == 1 else "✅ Low chance of Heart Disease"

        # -------------------------------------------
        # GRAPH 1: Probability Bar
        # -------------------------------------------
        plt.figure()
        plt.bar(["No Disease", "Disease"], [1 - probability, probability])
        plt.title("Risk Probability")
        plt.savefig("static/images/risk_probability.png")
        plt.close()

        # -------------------------------------------
        # GRAPH 2: Probability Pie Chart
        # -------------------------------------------
        plt.figure()
        plt.pie([1 - probability, probability],
                labels=["Safe", "At Risk"],
                autopct="%1.1f%%")
        plt.title("Risk Distribution")
        plt.savefig("static/images/pie_risk.png")
        plt.close()

        # -------------------------------------------
        # GRAPH 3: User Input Bar Graph
        # -------------------------------------------
        feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                         "restecg", "thalach", "exang", "oldpeak",
                         "slope", "ca", "thal"]

        plt.figure(figsize=(10, 4))
        plt.bar(feature_names, data)
        plt.xticks(rotation=90)
        plt.title("Patient Input Parameters")
        plt.tight_layout()
        plt.savefig("static/images/input_bar.png")
        plt.close()

        # -------------------------------------------
        # GRAPH 4: Custom Confusion Matrix (based on prediction)
        # -------------------------------------------
        # user_true = [prediction]  # assume model correct for confusion matrix
        # user_pred = [prediction]

        # cm = np.zeros((2, 2), dtype=int)
        # cm[prediction][prediction] += 1

        # plt.figure(figsize=(4, 3))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.title("Confusion Matrix (User Based)")
        # plt.savefig("static/images/confusion_matrix.png")
        # plt.close()

        # -------------------------------------------
        # GRAPH 5: ROC-like curve (using probability)
        # -------------------------------------------
        plt.figure()
        plt.plot([0, 1 - probability], [0, probability], marker='o')
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.savefig("static/images/roc_curve.png")
        plt.close()

        # -------------------------------------------
        # GRAPH 6: Random Feature Importance
        # -------------------------------------------
        plt.figure(figsize=(10, 4))
        plt.bar(feature_names, np.random.rand(len(feature_names)))
        plt.xticks(rotation=90)
        plt.title("Feature Importance (Sample)")
        plt.tight_layout()
        plt.savefig("static/images/feature_importance.png")
        plt.close()

        return render_template(
            "report.html",
            prediction_text=result,
            probability=round(probability * 100, 2),
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            user_name=session["username"]
        )

    return render_template("predict.html")


@app.route("/report")
def report_page():
    return render_template("report.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
