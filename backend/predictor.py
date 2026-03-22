from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


@dataclass
class InferenceArtifacts:
    vectorizer: Any
    feature_mask: np.ndarray
    svd: Any
    model: Any
    artifact_dir: Path


class SpamPredictor:
    def __init__(self) -> None:
        self.art = self._load_artifacts()

    def predict(self, text: str) -> Dict[str, Any]:
        text = self._normalize_input(text)

        if not text:
            return {
                "label": "ham",
                "probability": 0.0,
                "confidence": 0.0,
            }

        X = self.art.vectorizer.transform([text])

        mask = self.art.feature_mask
        if mask.dtype != bool:
            mask = mask.astype(bool)

        if mask.shape[0] != X.shape[1]:
            raise ValueError(
                f"Feature mask length ({mask.shape[0]}) does not match "
                f"TF-IDF feature size ({X.shape[1]})."
            )

        X_fs = X[:, mask]
        X_dense = self.art.svd.transform(X_fs).astype(np.float32)

        if hasattr(self.art.model, "predict_proba"):
            proba_spam = float(self.art.model.predict_proba(X_dense)[0, 1])
        else:
            pred = self.art.model.predict(X_dense)
            proba_spam = float(pred[0])

        proba_spam = max(0.0, min(1.0, proba_spam))
        label = "spam" if proba_spam >= 0.5 else "ham"
        confidence = proba_spam if label == "spam" else (1.0 - proba_spam)

        return {
            "label": label,
            "probability": round(proba_spam, 6),
            "confidence": round(confidence * 100, 2),
        }

    def _normalize_input(self, text: str) -> str:
        text = str(text or "").lower()
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\b\d+\b", " <NUM> ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_artifacts(self) -> InferenceArtifacts:
        base_dir = Path(__file__).resolve().parent
        artifact_dir = base_dir / "model"

        vectorizer_path = artifact_dir / "vectorizer.joblib"
        mask_path = artifact_dir / "feature_mask.npy"
        svd_path = artifact_dir / "svd.joblib"
        model_path = artifact_dir / "classifier.joblib"

        required_files = [vectorizer_path, mask_path, svd_path, model_path]
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                "Missing required model artifact(s): " + ", ".join(missing_files)
            )

        vectorizer = joblib.load(vectorizer_path)
        feature_mask = np.load(mask_path, allow_pickle=False)
        svd = joblib.load(svd_path)
        model = joblib.load(model_path)

        return InferenceArtifacts(
            vectorizer=vectorizer,
            feature_mask=feature_mask,
            svd=svd,
            model=model,
            artifact_dir=artifact_dir,
        )