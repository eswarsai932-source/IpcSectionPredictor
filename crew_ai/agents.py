from crewai import Agent
from crew_ai.llm import smart_llm  # ✅ your hybrid LLM


# ---------------------------------
# 🧾 Explanation Agent
# ---------------------------------
explanation_agent = Agent(
    role="Legal Explanation Agent",
    goal="Explain IPC sections strictly based on provided data",
    backstory="Expert in Indian Penal Code with zero hallucination tolerance",
    llm=smart_llm  # ✅ FIXED
)


# ---------------------------------
# 🕵️ Crime Extraction Agent
# ---------------------------------
crime_agent = Agent(
    role="Crime Fact Extraction Agent",
    goal="Extract legally relevant facts from crime description",
    backstory="Expert in Indian criminal law and legal fact extraction",
    llm=smart_llm  # ✅ FIXED
)


# ---------------------------------
# ⚖️ IPC Analysis Agent
# ---------------------------------
ipc_agent = Agent(
    role="IPC Section Analysis Agent",
    goal="Analyze predicted IPC sections and determine applicability",
    backstory="Expert in Indian Penal Code and legal reasoning",
    llm=smart_llm  # ✅ FIXED
)