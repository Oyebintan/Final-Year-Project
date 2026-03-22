import os
import secrets
from pathlib import Path

import pandas as pd
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
df = None
ham_texts = []
spam_texts = []
_last_sample = {"ham": None, "spam": None}


def get_predictor() -> SpamPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpamPredictor()
    return _predictor


def load_dataset() -> None:
    global df, ham_texts, spam_texts

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH} (sample endpoint will not work).")
        df = None
        ham_texts = []
        spam_texts = []
        return

    data = pd.read_csv(DATA_PATH).dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(int)

    df = data
    ham_texts = df[df["label"] == 0]["text"].astype(str).tolist()
    spam_texts = df[df["label"] == 1]["text"].astype(str).tolist()

    print(f"Dataset loaded successfully: ham={len(ham_texts)}, spam={len(spam_texts)}")


def pick_random(label: str) -> str:
    pool = ham_texts if label == "ham" else spam_texts
    if not pool:
        return ""

    last = _last_sample[label]

    for _ in range(8):
        text = pool[secrets.randbelow(len(pool))]
        if text != last:
            _last_sample[label] = text
            return text

    text = pool[secrets.randbelow(len(pool))]
    _last_sample[label] = text
    return text


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Email Spam Classification API is running.",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "sample_ham": "/sample?label=ham",
                "sample_spam": "/sample?label=spam",
            },
        }
    )


@app.get("/health")
def health():
    try:
        predictor_ready = False
        dataset_ready = DATA_PATH.exists()

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
                "dataset_ready": dataset_ready,
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


@app.get("/sample")
def sample():
    try:
        label = str(request.args.get("label") or "").strip().lower()

        if label not in ("ham", "spam"):
            return jsonify({"error": "Use /sample?label=ham or /sample?label=spam"}), 400

        if df is None:
            return jsonify({"error": "Dataset not loaded. Ensure backend/spam.csv exists."}), 500

        text = pick_random(label)
        if not text:
            return jsonify({"error": f"No samples available for {label}"}), 500

        return jsonify({"label": label, "text": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


load_dataset()


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()