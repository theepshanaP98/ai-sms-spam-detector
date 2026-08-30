from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = MODEL_DIR / "spam_classifier.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

UCI_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip",
]


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        print(f"Dataset already exists: {DATA_FILE}")
        return

    last_error = None
    for url in UCI_URLS:
        try:
            print(f"Downloading UCI SMS Spam Collection from {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                members = archive.namelist()
                target = next((m for m in members if m.endswith("SMSSpamCollection")), None)
                if not target:
                    raise RuntimeError("SMSSpamCollection file was not found in the downloaded archive.")
                DATA_FILE.write_bytes(archive.read(target))
            print(f"Saved dataset to {DATA_FILE}")
            return
        except Exception as exc:  # tries mirror URL before failing
            last_error = exc

    raise RuntimeError(
        "Unable to download the UCI dataset automatically. Download the SMS Spam Collection manually "
        "and place the file named 'SMSSpamCollection' inside the data/ directory."
    ) from last_error


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )


def train() -> None:
    download_dataset()
    df = load_dataset().dropna().drop_duplicates().reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", pos_label="spam", zero_division=0
    )

    metrics = {
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "spam_precision": round(float(precision), 4),
        "spam_recall": round(float(recall), 4),
        "spam_f1": round(float(f1), 4),
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "random_state": 42,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    train()
