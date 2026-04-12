    from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import tensorflow as tf


@dataclass
class InferenceArtifacts:
    feature_pipeline: Any
    l1_selector: Any
    label_encoder: Any
    model: tf.keras.Model
    artifact_dir: Path


def build_deep_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


class SpamPredictor:
    def __init__(self, artifact_dir: str | None = None) -> None:
        self.art = self._load_artifacts(artifact_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        text = self._normalize_input(text)

        if not text:
            return {
                "label": "ham",
                "probability": 0.0,
                "confidence": 0.0,
            }

        x = self.art.feature_pipeline.transform([text])

        if hasattr(x, "toarray"):
            x = x.toarray().astype(np.float32)
        else:
            x = np.asarray(x, dtype=np.float32)

        x_selected = self.art.l1_selector.transform(x).astype(np.float32)

        proba_spam = float(self.art.model.predict(x_selected, verbose=0).ravel()[0])
        proba_spam = max(0.0, min(1.0, proba_spam))

        label = "spam" if proba_spam >= 0.5 else "ham"
        confidence = proba_spam if label == "spam" else (1.0 - proba_spam)

        return {
            "label": label,
            "probability": round(proba_spam, 6),
            "confidence": round(confidence * 100, 2),
        }

    def _normalize_input(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_artifacts(self, artifact_dir: str | None) -> InferenceArtifacts:
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        out_dir = Path(artifact_dir) if artifact_dir else (project_root / "outputs_dl")

        pipeline_path = out_dir / "pipeline.pkl"
        weights_path = out_dir / "model.weights.h5"

        required_files = [pipeline_path, weights_path]
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                "Missing required artifact(s): " + ", ".join(missing_files)
            )

        pipeline_obj = joblib.load(pipeline_path)

        feature_pipeline = pipeline_obj.get("feature_pipeline")
        l1_selector = pipeline_obj.get("l1_selector")
        label_encoder = pipeline_obj.get("label_encoder")

        if feature_pipeline is None or l1_selector is None:
            raise ValueError("pipeline.pkl is missing required preprocessing objects.")

        dummy = feature_pipeline.transform(["test message"])
        if hasattr(dummy, "toarray"):
            dummy = dummy.toarray().astype(np.float32)
        else:
            dummy = np.asarray(dummy, dtype=np.float32)

        dummy_selected = l1_selector.transform(dummy).astype(np.float32)
        input_dim = dummy_selected.shape[1]

        model = build_deep_model(input_dim)
        model.load_weights(weights_path)

        return InferenceArtifacts(
            feature_pipeline=feature_pipeline,
            l1_selector=l1_selector,
            label_encoder=label_encoder,
            model=model,
            artifact_dir=out_dir,
        )