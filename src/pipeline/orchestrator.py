from llm.backend import send_prompt

def run_pipeline(text: str) -> str:
    return send_prompt(text)