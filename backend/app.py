import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .predictor import SpamPredictor
except ImportError:
    from predictor import SpamPredictor


app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"

_predictor = None


def get_predictor() -> SpamPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpamPredictor()
    return _predictor


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Email Spam Classification API is running.",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
            },
        }
    )


@app.get("/health")
def health():
    try:
        predictor_ready = False

        try:
            get_predictor()
            predictor_ready = True
        except Exception:
            predictor_ready = False

        return jsonify(
            {
                "status": "ok",
                "service": "email-spam-classifier",
                "predictor_ready": predictor_ready,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "").strip()

        if not text:
            return jsonify({"error": "Field 'text' is required."}), 400

        result = get_predictor().predict(text)
        probability = float(result.get("probability", 0.0))
        label = result.get("label", "ham")
        confidence = float(result.get("confidence", 0.0))

        return jsonify(
            {
                "label": label,
                "probability": probability,
                "spam_probability": probability,
                "ham_probability": round(1.0 - probability, 6),
                "confidence": confidence,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()