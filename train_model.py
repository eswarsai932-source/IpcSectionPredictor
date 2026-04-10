import joblib
import numpy as np

# Load trained artifacts (once)
model = joblib.load("ipc_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")


def predict_ipc_with_confidence(crime_text, top_k=3, threshold=0.3):
    """
    Predict IPC sections with confidence scores
    """

    if not crime_text.strip():
        return []

    # Vectorize input
    X = vectorizer.transform([crime_text])

    # Predict probabilities
    probs = model.predict_proba(X)[0]

    # Sort top-k
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = [
        (mlb.classes_[idx], round(float(probs[idx]), 3))
        for idx in top_indices if probs[idx] >= threshold
    ]

    # Fallback (always return at least one)
    if not results:
        best_idx = np.argmax(probs)
        results = [
            (mlb.classes_[best_idx], round(float(probs[best_idx]), 3))
        ]

    return results