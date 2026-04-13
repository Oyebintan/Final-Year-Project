import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

out_dir = Path("outputs_dl")
pipeline_obj = joblib.load(out_dir / "pipeline.pkl")
feature_pipeline = pipeline_obj["feature_pipeline"]
l1_selector = pipeline_obj["l1_selector"]

df = pd.read_csv("backend/spam.csv")
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(int)

X = feature_pipeline.transform(df["text"].tolist())
X = X.toarray()
X_sel = l1_selector.transform(X)
y = df["label"].values

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_sel, y)
print(classification_report(y, clf.predict(X_sel)))

pipeline_obj["classifier"] = clf
joblib.dump(pipeline_obj, out_dir / "pipeline.pkl")
print("Saved!")
