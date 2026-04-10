import joblib
import numpy as np


# ======================================================
# Load trained artifacts ONCE
# ======================================================
model = joblib.load("ipc_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")


# ======================================================
# 🔹 Preprocessing
# ======================================================
def preprocess(text):
    text = text.lower()

    replacements = {
        "harassing": "harass",
        "molesting": "molest",
        "threatening": "threat",
        "stalking": "stalk",
        "beating": "assault",
        "killed": "murder",
        "stolen": "steal"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# ======================================================
# 🔹 Rule Engine (IMPORTANT)
# ======================================================
def rule_engine(text):
    text = text.lower()

    if "harass" in text or "molest" in text:
        return ["354", "509"]

    if "threat" in text:
        return ["506"]

    if "steal" in text or "theft" in text:
        return ["379"]

    return []


# ======================================================
# 🔹 ML Prediction
# ======================================================
def predict_ipc_with_confidence(crime_text, top_k=3):
    text = preprocess(crime_text)

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    top_indices = probs.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append(
            (str(mlb.classes_[idx]), round(float(probs[idx]), 3))
        )

    return results


# ======================================================
# 🔥 FINAL HYBRID PREDICTION (IMPORTANT)
# ======================================================
def final_prediction(crime_text):
    text = preprocess(crime_text)

    rule_preds = rule_engine(text)
    ml_preds = predict_ipc_with_confidence(text)

    final = []

    # Priority to rules
    for r in rule_preds:
        final.append((r, 0.95))

    # Add ML predictions
    for ipc, score in ml_preds:
        if ipc not in [f[0] for f in final]:
            final.append((ipc, score))

    return final[:3]
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_cases(text, df, vectorizer, top_n=3):
    text_vec = vectorizer.transform([text])
    dataset_vec = vectorizer.transform(df['complaint_text'])

    sims = cosine_similarity(text_vec, dataset_vec)[0]

    df['similarity'] = sims

    return df.sort_values(by='similarity', ascending=False).head(top_n)