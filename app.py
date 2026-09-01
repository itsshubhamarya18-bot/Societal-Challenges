from flask import Flask, render_template, request
from src.predict import predict_problem

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    problem_text = request.form.get("problem_text", "").strip()

    if not problem_text:
        return render_template(
            "index.html",
            error="Please enter a problem description."
        )

    try:
        result = predict_problem(problem_text)

        return render_template(
            "results.html",
            problem_text=problem_text,
            result=result
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )