from pathlib import Path
from functools import lru_cache
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.joblib"

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python scripts/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_message(message: str) -> dict:
    text = message.strip()
    if not text:
        raise ValueError("Message cannot be empty.")

    model = load_model()
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    spam_probability = float(probabilities[classes.index("spam")])
    ham_probability = float(probabilities[classes.index("ham")])

    return {
        "label": str(prediction),
        "spam_probability": round(spam_probability, 4),
        "ham_probability": round(ham_probability, 4),
        "confidence": round(max(spam_probability, ham_probability), 4),
    }
