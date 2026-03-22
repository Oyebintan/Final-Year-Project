import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model")

VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.joblib")
FILTER_SELECTOR_PATH = os.path.join(ARTIFACT_DIR, "filter_selector.joblib")
FEATURE_MASK_PATH = os.path.join(ARTIFACT_DIR, "feature_mask.npy")
SVD_PATH = os.path.join(ARTIFACT_DIR, "svd.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "classifier.joblib")

CONF_MATRIX_PATH = os.path.join(ARTIFACT_DIR, "confusion_matrix.csv")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "classification_report.json")

TFIDF_MAX_FEATURES = 50000
CHI2_KBEST = 15000
SVD_COMPONENTS = 250


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\b\d+\b", " <NUM> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_artifacts_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def train() -> None:
    print("Training started...")

    ensure_artifacts_dir()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)

    print(f"Dataset shape: {df.shape}")
    print(df["label"].value_counts())

    y = df["label"].values
    X_text = df["text"].values

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2
    )

    print("Fitting TF-IDF...")
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)

    k_best = min(CHI2_KBEST, X_train.shape[1])
    filter_sel = SelectKBest(score_func=chi2, k=k_best)
    filter_sel.fit(X_train, y_train)

    feature_mask = filter_sel.get_support(indices=False).astype(bool)

    X_train_fs = X_train[:, feature_mask]
    X_test_fs = X_test[:, feature_mask]

    n_comp = min(SVD_COMPONENTS, X_train_fs.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)

    print("Applying SVD...")
    X_train_dense = svd.fit_transform(X_train_fs).astype(np.float32)
    X_test_dense = svd.transform(X_test_fs).astype(np.float32)

    print("Training classifier...")
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X_train_dense, y_train)

    probs = clf.predict_proba(X_test_dense)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    cm_df = pd.DataFrame(
        cm,
        columns=["Pred_Ham", "Pred_Spam"],
        index=["True_Ham", "True_Spam"]
    )
    cm_df.to_csv(CONF_MATRIX_PATH)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("Saving artifacts...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(filter_sel, FILTER_SELECTOR_PATH)
    np.save(FEATURE_MASK_PATH, feature_mask)
    joblib.dump(svd, SVD_PATH)
    joblib.dump(clf, MODEL_PATH)

    print("\nTraining complete.")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train()