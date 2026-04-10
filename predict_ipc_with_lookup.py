import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
# 🔹 Rule Engine
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
def predict_ipc_with_confidence(text, model, vectorizer, mlb, top_k=3):
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
# 🔥 FINAL FUNCTION (THIS FIXES YOUR ERROR)
# ======================================================
def predict_ipc(crime_text):
    # ✅ Load models INSIDE function (important)
    model = joblib.load("ipc_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    mlb = joblib.load("label_binarizer.pkl")

    text = preprocess(crime_text)

    rule_preds = rule_engine(text)
    ml_preds = predict_ipc_with_confidence(text, model, vectorizer, mlb)

    final = []

    # Priority to rules
    for r in rule_preds:
        final.append((r, 0.95))

    # Add ML predictions
    for ipc, score in ml_preds:
        if ipc not in [f[0] for f in final]:
            final.append((ipc, score))

    final = final[:3]

    # Convert to app.py format
    if final:
        return {
            "section": final[0][0],
            "confidence": final[0][1],
            "description": f"Predicted IPC Sections: {', '.join([f[0] for f in final])}"
        }
    else:
        return {
            "section": "N/A",
            "confidence": 0.0,
            "description": "No prediction available"
        }


# ======================================================
# 🔹 Similar Cases
# ======================================================
def get_similar_cases(text, df, vectorizer, top_n=3):
    text_vec = vectorizer.transform([text])
    dataset_vec = vectorizer.transform(df['complaint_text'])

    sims = cosine_similarity(text_vec, dataset_vec)[0]

    df['similarity'] = sims

    return df.sort_values(by='similarity', ascending=False).head(top_n)
