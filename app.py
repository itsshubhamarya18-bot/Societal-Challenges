from flask import Flask, render_template, request, jsonify
from src.predict import predict_problem

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# Website prediction route
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


# JSON API route for backend developers
@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    if not data or "problem_text" not in data:
        return jsonify({
            "error": "problem_text is required"
        }), 400

    problem_text = data["problem_text"].strip()

    if not problem_text:
        return jsonify({
            "error": "Problem description cannot be empty."
        }), 400

    try:
        result = predict_problem(problem_text)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )