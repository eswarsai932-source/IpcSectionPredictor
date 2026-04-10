import requests
import os


# ---------------------------------
# 🌐 API LLM (Qwen)
# ---------------------------------
def get_api_llm(prompt):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "qwen/qwen3.5-flash-02-23",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            return f"⚠️ API Error: {response.text}"

        result = response.json()
        return result['choices'][0]['message']['content']

    except Exception as e:
        return f"⚠️ Exception: {str(e)}"


# ---------------------------------
# 🧠 HYBRID LLM (Local + API)
# ---------------------------------
def smart_llm(prompt):

    # 🔹 Try local (Mistral via Ollama)
    try:
        import ollama
        res = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}]
        )

        if res and 'message' in res:
            return res['message']['content'] + "\n\n✅ Local LLM"

    except Exception:
        pass  # silently fallback

    # 🔹 Fallback to API (Qwen)
    return get_api_llm(prompt) + "\n\n🌐 Qwen API"