import ollama
from config import MODEL_NAME

def send_prompt(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert software engineer"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
    except Exception as e:
        return f"Error while calling model:{e}"