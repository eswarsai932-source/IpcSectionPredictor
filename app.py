import os
os.environ["OPENAI_API_KEY"] = "sk-dummy"

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
# STYLE (SAFE CSS ONLY)
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
        explanation = result.get("description", "No explanation available")

        cases = get_similar_cases(complaint)

    tab1, tab2, tab3 = st.tabs(["⚖️ IPC Sections", "📚 Similar Cases", "👨‍⚖️ Lawyers"])

    # ---------------- TAB 1 ----------------
    with tab1:

        for ipc, score in preds:

            st.subheader(f"⚖️ IPC Section {ipc}")

            # ✅ SAFE progress bar (no HTML)
            st.progress(score)

            st.success(f"Confidence: {score*100:.1f}%")

            st.markdown("**📖 Description:**")
            st.write(explanation)

            st.markdown("**🧾 Simple Explanation:**")
            st.write(explanation)

            st.markdown("**⚖️ Punishment:**")
            st.info("Information currently unavailable")

            st.divider()

        st.subheader("🧾 Legal Explanation")
        st.info(explanation)

        report = generate_report(preds, explanation)
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
