from crewai import Crew
from crew_ai.tasks import legal_explanation_task
from crew_ai.agents import explanation_agent
from predict_ipc_with_lookup import predict_ipc_with_confidence, final_prediction
from crew_ai.llm import smart_llm
from predict_ipc_with_lookup import get_similar_cases, vectorizer

# -------------------------------
# 1. CREWAI FUNCTION (optional)
# -------------------------------
def run_crew(crime_text):
    raw_predictions = predict_ipc_with_confidence(crime_text)

    ipc_predictions_safe = [
        {"section": sec, "confidence": score}
        for sec, score in raw_predictions
    ]

    crew = Crew(
        agents=[explanation_agent],
        tasks=[legal_explanation_task],
        verbose=False
    )

    result = crew.kickoff(
        inputs={
            "crime_description": crime_text,
            "predicted_ipc_sections": ipc_predictions_safe
        }
    )

    return ipc_predictions_safe, result


# -------------------------------
# 2. HYBRID PIPELINE (MAIN)
# -------------------------------
def run_pipeline(text, df):
    preds = final_prediction(text)
    ipc_list = [ipc for ipc, _ in preds]

    # 🔹 Main explanation
    prompt = f"""
    Complaint: {text}
    IPC Sections: {ipc_list}
    Explain clearly in simple terms.
    """
    explanation = smart_llm(prompt)

    # 🔹 Similar cases
    cases = get_similar_cases(text, df, vectorizer)

    # 🔥 LLM IPC DETAILS
    ipc_details = {}

    import json

    for ipc, score in preds:
        ipc = str(ipc)

        detail_prompt = f"""
        Give details for IPC Section {ipc} in STRICT JSON format:

        {{
        "description": "...",
        "simple_explanation": "...",
        "punishment": "..."
        }}

        No extra text.
        """

        try:
            response = smart_llm(detail_prompt)

            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]

            data = json.loads(json_str)

            ipc_details[ipc] = {
                "description": data.get("description", ""),
                "simple_description": data.get("simple_explanation", ""),
                "punishment": data.get("punishment", "")
            }

        except Exception:
            ipc_details[ipc] = {
                "description": "Not available",
                "simple_description": "Not available",
                "punishment": "Not available"
            }

    return preds, explanation, cases, ipc_details