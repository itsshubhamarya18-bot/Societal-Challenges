from flask import Flask, request, jsonify, render_template

from src.predict import predict_problem


app = Flask(__name__)


# ============================================================
# PAGE 1 - INPUT
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PAGE 2 - RESULTS
# ============================================================

@app.route("/results")
def results():
    return render_template("results.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        problem_text = data.get("problem_text")

        if not problem_text or not problem_text.strip():
            return jsonify({
                "error": "Problem text is required."
            }), 400

        result = predict_problem(problem_text)

        return jsonify(result)

    except ValueError as e:

        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )