# Hybrid Feature Selection for Email Spam Classification using Deep Learning

This project provides an end-to-end implementation of a **hybrid feature selection pipeline** for classifying emails as **spam** or **ham**.

## What makes it "hybrid"?

The feature selection strategy combines two stages:

1. **Filter method**: Chi-Square (`chi2`) test on TF-IDF features to keep statistically relevant terms.
2. **Embedded method**: L1-regularized Logistic Regression to remove weak/irrelevant features by enforcing sparse coefficients.

The selected feature subset is then fed into a deep neural network classifier.

## Project Structure

- `spam_hybrid_dl.py` – training script and reusable pipeline.
- `requirements.txt` – Python package requirements.

## Dataset format

Prepare a CSV file with at least these columns:

- `text` : email content
- `label`: spam/ham target (e.g., `spam`, `ham`, `1`, `0`)

Example:

```csv
text,label
"Free entry in 2 a wkly comp to win FA Cup final tkts",spam
"Hey, are we still meeting this evening?",ham
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python spam_hybrid_dl.py \
  --data_path path/to/emails.csv \
  --text_col text \
  --label_col label \
  --max_features 20000 \
  --chi2_k 5000 \
  --hidden_dims 256 128 64 \
  --dropout 0.3 \
  --epochs 12 \
  --batch_size 64 \
  --output_dir outputs
```

## Output

The script writes:

- `metrics.json` – accuracy, precision, recall, F1, ROC-AUC.
- `report.txt` – full classification report.
- `confusion_matrix.npy` – confusion matrix.
- `pipeline.pkl` – fitted vectorizer + selectors + label encoder.
- `model.keras` or `model.pkl` – trained deep model.

## Notes for final year project write-up

You can describe the contribution as:

- **A two-stage hybrid feature selector** reducing high-dimensional TF-IDF noise.
- **A deep classifier** trained on compact, discriminative features.
- **Empirical comparison** with and without feature selection to show performance and efficiency gains.

