import os
os.environ["OPENAI_API_KEY"] = "sk-dummy"

import streamlit as st
import joblib
import pandas as pd
from crew_ai.crew_runner import run_pipeline

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="IPC Intelligence System",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------
# SAFE HTML RENDER FUNCTION (🔥 FIX)
# -------------------------------------------------
def render_html(html):
    st.markdown(html, unsafe_allow_html=True)

# -------------------------------------------------
# UI STYLE (UNCHANGED)
# -------------------------------------------------
render_html("""
<style>
body { background-color: #f5f7fb; }

.header {
    text-align: center;
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: white;
    margin-bottom: 25px;
}

.card {
    background: #3e3f40;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: ivory;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.section-title {
    color: #2563eb;
    font-size: 18px;
    margin-bottom: 10px;
}

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

.empty-text {
    color: #6b7280;
    font-style: italic;
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
    return pd.read_csv("ipc_training_dataset.csv")

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

# -------------------------------------------------
# HEADER
# -------------------------------------------------
render_html("""
<div class="header">
    <h1>⚖️ IPC Intelligence System</h1>
</div>
""")

# -------------------------------------------------
# INPUT
# -------------------------------------------------
complaint = st.text_area("📝 Enter Complaint", height=150)

# -------------------------------------------------
# BUTTON
# -------------------------------------------------
if st.button("🚀 Analyze Complaint"):

    if not complaint.strip():
        st.warning("Please enter complaint text.")
        st.stop()

    with st.spinner("🧠 AI analyzing..."):
        preds, explanation, cases, ipc_details = run_pipeline(complaint, cases_df)

    tab1, tab2, tab3 = st.tabs(["⚖️ IPC Sections", "📚 Cases", "👨‍⚖️ Lawyers"])

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

            description = details.get("description", "").strip()
            simple_exp = details.get("simple_description", "").strip()
            punishment = details.get("punishment", "").strip()

            if not description or description.lower() in ["n/a", "not available"]:
                description = "<span class='empty-text'>Information currently unavailable</span>"

            if not simple_exp or simple_exp.lower() in ["n/a", "not available"]:
                simple_exp = "<span class='empty-text'>Information currently unavailable</span>"

            if not punishment or punishment.lower() in ["n/a", "not available"]:
                punishment = "<span class='empty-text'>Information currently unavailable</span>"

            render_html(f"""
            <div class="card">
                <div class="section-title">IPC Section {ipc}</div>
                <div class="bar">
                    <div class="{bar_class}" style="width:{score*100}%; height:10px;"></div>
                </div>
                <p><b>Confidence:</b> {score*100:.1f}%</p>
                <p><b>📖 Description:</b><br>{description}</p>
                <p><b>🧾 Simple Explanation:</b><br>{simple_exp}</p>
                <p><b>⚖️ Punishment:</b><br>{punishment}</p>
            </div>
            """)

        render_html(f"<h3>🧾 Legal Explanation</h3>")
        render_html(f"<div class='card'>{explanation}</div>")

        report = generate_report(preds, explanation)
        st.download_button("📄 Download Report", report, "ipc_report.txt")

    # ---------------- TAB 2 ----------------
    with tab2:

        render_html("<h3>📚 Similar Cases</h3>")

        if cases is not None and not cases.empty:
            for _, row in cases.iterrows():
                render_html(f"""
                <div class="card">
                    <b>Similarity:</b> {row.get('similarity', 0):.2f}<br><br>
                    {row.get('complaint_text', '')}<br><br>
                    <b>IPC:</b> {row.get('ipc_sections', '')}
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

    # Footer
    render_html("""
    <div style="text-align:center;margin-top:20px">
        <p style="color:#94a3b8">⚠️ Educational purpose only</p>
        <p style="color:#22c55e">🚀 Hybrid AI Active</p>
    </div>
    """)