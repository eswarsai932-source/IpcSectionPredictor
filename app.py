import os
import requests
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from predict_ipc_with_lookup import predict_ipc

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="IPC Intelligence System",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
.header {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(90deg, #1e3a8a, #38bdf8);
    color: white;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_resource
def load_models():
    return (
        joblib.load("ipc_model.pkl"),
        joblib.load("tfidf_vectorizer.pkl"),
        joblib.load("label_binarizer.pkl")
    )

@st.cache_data
def load_cases():
    df = pd.read_csv("ipc_training_dataset.csv")
    df["ipc_sections"] = df["ipc_sections"].astype(str).str.strip(",")
    return df

@st.cache_data
def load_lawyers():
    return pd.read_csv("lawyers.csv")

model, vectorizer, mlb = load_models()
cases_df = load_cases()
lawyers_df = load_lawyers()

# -------------------------------------------------
# ✅ FIXED OPENROUTER FUNCTION
# -------------------------------------------------
def get_llm_explanation(ipc, complaint):
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app-name.streamlit.app",
                "X-Title": "IPC Intelligence System"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal assistant for Indian Penal Code."
                    },
                    {
                        "role": "user",
                        "content": f"""
Explain IPC Section {ipc} clearly.

Complaint:
{complaint}

Give:
Description:
Simple Explanation:
Punishment:
"""
                    }
                ],
                "temperature": 0.3
            }
        )

        data = response.json()

        # 🔥 Show actual error if exists
        if "error" in data:
            return f"❌ API Error: {data['error']}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Exception: {str(e)}"

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------
def recommend_lawyers(predicted):
    mapping = {
        "379": "Criminal Lawyer",
        "506": "Criminal Lawyer",
        "420": "Fraud Lawyer",
        "498A": "Family Lawyer",
        "354": "Women Safety Lawyer"
    }
    needed = {mapping.get(sec) for sec, _ in predicted if sec in mapping}
    return lawyers_df[lawyers_df["specialization"].isin(needed)].head(5)

def generate_report(preds, explanation):
    report = "IPC Prediction Report\n\n"
    for ipc, score in preds:
        report += f"IPC {ipc} - {score*100:.1f}%\n"
    report += "\nExplanation:\n" + explanation
    return report

def get_similar_cases(user_text, top_k=5):
    try:
        user_vec = vectorizer.transform([user_text])
        case_vecs = vectorizer.transform(cases_df["complaint_text"])

        sims = cosine_similarity(user_vec, case_vecs).flatten()
        cases_df["similarity"] = sims

        results = cases_df.sort_values(by="similarity", ascending=False).head(top_k)
        results = results[results["similarity"] > 0.2]

        return results
    except:
        return pd.DataFrame()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="header">
    <h1>⚖️ IPC Intelligence System</h1>
    <p>AI-powered legal complaint analyzer</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT
# -------------------------------------------------
complaint = st.text_area(
    "📝 Enter Complaint",
    placeholder="Describe the incident in detail...",
    height=150
)

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🚀 Analyze Complaint"):

    if not complaint.strip():
        st.warning("Please enter complaint text.")
        st.stop()

    with st.spinner("🧠 AI analyzing complaint..."):

        result = predict_ipc(complaint)
        preds = [(result.get("section", "N/A"), result.get("confidence", 0.0))]
        cases = get_similar_cases(complaint)

    tab1, tab2, tab3 = st.tabs(["⚖️ IPC Sections", "📚 Similar Cases", "👨‍⚖️ Lawyers"])

    # ---------------- TAB 1 ----------------
    with tab1:

        for ipc, score in preds:
            ipc = str(ipc).strip()

            llm_explanation = get_llm_explanation(ipc, complaint)

            if score >= 0.8:
                color = "#22c55e"
            elif score >= 0.5:
                color = "#f59e0b"
            else:
                color = "#ef4444"

            st.markdown(f"""
<div style="
    background:#2f2f2f;
    padding:20px;
    border-radius:14px;
    margin-bottom:20px;
    color:white;
">

<h4 style="color:#3b82f6;">⚖️ IPC Section {ipc}</h4>

<div style="
    width:100%;
    background:#d1d5db;
    height:10px;
    border-radius:10px;
    margin:10px 0;
">
    <div style="
        width:{score*100}%;
        background:{color};
        height:10px;
        border-radius:10px;
    "></div>
</div>

<p><b>Confidence:</b> {score*100:.1f}%</p>

<p><b>🤖 AI Explanation:</b><br>{llm_explanation}</p>

</div>
""", unsafe_allow_html=True)

        report = generate_report(preds, llm_explanation)
        st.download_button("📄 Download Report", report, "ipc_report.txt")

    # ---------------- TAB 2 ----------------
    with tab2:

        st.subheader("📚 Similar Cases")

        if not cases.empty:
            for _, row in cases.iterrows():
                st.markdown(f"""
**Similarity:** {row.get('similarity', 0):.2f}

{row.get('complaint_text', '')}

**IPC Sections:** {row.get('ipc_sections', '')}
""")
                st.divider()
        else:
            st.info("No similar cases found.")

    # ---------------- TAB 3 ----------------
    with tab3:

        st.subheader("👨‍⚖️ Recommended Lawyers")

        lawyers = recommend_lawyers(preds)

        if lawyers.empty:
            st.info("No recommendations available.")
        else:
            st.dataframe(lawyers, use_container_width=True)

    st.markdown(
        "<p style='text-align:center;color:gray;'>⚠️ Educational purpose only</p>",
        unsafe_allow_html=True
    )
