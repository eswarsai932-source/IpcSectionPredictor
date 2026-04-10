from crewai import Task
from crew_ai.agents import crime_agent, ipc_agent, explanation_agent

crime_task = Task(
    description="Extract key legal facts from the crime description.",
    expected_output="Bullet-point list of legally relevant facts.",
    agent=crime_agent
)
ipc_task = Task(
    description=(
        "You are given multiple predicted IPC sections with confidence scores. "
        "Analyze which IPC sections are legally applicable."
    ),
    expected_output="Reasoned evaluation of each IPC section and final applicable sections.",
    agent=ipc_agent
)
legal_explanation_task = Task(
    description=(
        "You are a legal expert in Indian criminal law.\n\n"
        "Crime Description:\n"
        "{crime_description}\n\n"
        "Predicted IPC Sections:\n"
        "{predicted_ipc_sections}\n\n"
        "Your job is to:\n"
        "1. Explain what crime has occurred in simple language\n"
        "2. Explain which IPC sections apply and why\n"
        "3. Explain what legal action the victim can take (FIR, police, court)\n"
        "4. Explain the punishment under law\n\n"
        "Give a clear, victim-friendly legal explanation."
    ),
    expected_output="Clear legal explanation and legal procedure steps.",
    agent=explanation_agent
)

