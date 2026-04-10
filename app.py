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
# HTML RENDER
# -------------------------------------------------
def render_html(html):
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------
# 🌟 IMPROVED UI STYLE
# -------------------------------------------------
render_html("""
<style>
body { background-color: #f5f7fb; }

.header {
    text-align: center;
    padding: 25px;
    border-radius: 14px;
    background: linear-gradient(90deg, #1e3a8a, #38bdf8);
    color: white;
    margin-bottom: 25px;
}

/* CLEAN WHITE CARDS */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    color: #111827;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}

/* TITLES */
.section-title {
    color: #1e40af;
    font-size: 20px;
    font-weight: bold;
}

/* PROGRESS BAR */
.bar {
    height: 10px;
    background: #e5e7eb;
    border-radius: 6px;
    overflow: hidden;
    margin: 10px 0;
}

.bar-green { background: linear-gradient(90deg, #16a34a, #4ade80); }
.bar-yellow { background: linear-gradient(90deg, #f59e0b, #facc15); }
.bar-red { background: linear-gradient(90deg, #dc2626, #f87171); }

/* TEXT COLORS */
.confidence { color: #16a34a; font-weight: bold; }
.muted { color: #6b7280; }

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""")

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
render_html("""
<div class="header">
    <h1>⚖️ IPC Intelligence System</h1>
    <p>AI-powered legal complaint analyzer</p>
</div>
""")

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
        ipc_details = {}

    tab1, tab2, tab3 = st.tabs(["⚖️ IPC Sections", "📚 Similar Cases", "👨‍⚖️ Lawyers"])

    # ---------------- TAB 1 ----------------
    with tab1:

        for ipc, score in preds:
            ipc = str(ipc).strip()
            details = ipc_details.get(ipc, {})

            if score >= 0.8:
                bar_class = "bar-green"
            elif score >= 0.5:
                bar_class = "bar-yellow"
            else:
                bar_class = "bar-red"

            description = details.get("description", explanation)
            simple_exp = details.get("simple_description", explanation)
            punishment = details.get("punishment", "Information currently unavailable")

            render_html(f"""
            <div class="card">
                <div class="section-title">⚖️ IPC Section {ipc}</div>

                <div class="bar">
                    <div class="{bar_class}" style="width:{score*100}%; height:10px;"></div>
                </div>

                <p class="confidence">Confidence: {score*100:.1f}%</p>

                <p><b>📖 Description:</b><br>{description}</p>
                <p><b>🧾 Simple Explanation:</b><br>{simple_exp}</p>
                <p><b>⚖️ Punishment:</b><br>{punishment}</p>
            </div>
            """)

        render_html("<h3>🧾 Legal Explanation</h3>")
        render_html(f"<div class='card'>{explanation}</div>")

        report = generate_report(preds, explanation)
        st.download_button("📄 Download Report", report, "ipc_report.txt")

    # ---------------- TAB 2 ----------------
    with tab2:

        render_html("<h3>📚 Similar Cases</h3>")

        if not cases.empty:
            for _, row in cases.iterrows():
                render_html(f"""
                <div class="card">
                    <p><b>Similarity:</b> {row.get('similarity', 0):.2f}</p>
                    <p>{row.get('complaint_text', '')}</p>
                    <p><b>IPC Sections:</b> {row.get('ipc_sections', '')}</p>
                </div>
                """)
        else:
            st.info("No similar cases found.")

    # ---------------- TAB 3 ----------------
    with tab3:

        lawyers = recommend_lawyers(preds)

        if lawyers.empty:
            st.info("No recommendations available.")
        else:
            st.dataframe(lawyers, use_container_width=True)

    render_html("""
    <div style="text-align:center;margin-top:20px">
        <p class="muted">⚠️ Educational purpose only</p>
    </div>
    """)
