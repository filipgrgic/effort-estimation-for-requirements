import ollama
from config import MODEL_NAME


def send_to_model(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
    except Exception as e:
        return f"Error while calling model:{e}"


def extract_prompt(text: str) -> str:
    return send_to_model(
        f"""
You are an assistant that extracts software requirements from text.

Task:
Extract all software requirements from the text below.

Output format:
Return a valid JSON array containing the extracted requirements.
Your entire response must be valid JSON only. Do not include explanations or additional text.

Each requirement must follow this structure:

{{
  "description": string,
  "type": "functional" | "non_functional" | "constraint"
}}

Example output:

[
  {{
    "description": "The system shall allow users to log in using email and password.",
    "type": "functional"
  }}
]

Rules:
- Extract every requirement mentioned in the text.
- Each requirement must be a separate JSON object.
- The "description" should contain the full requirement.
- The "type" must be exactly one of: "functional", "non_functional", "constraint".
- If no requirements are found, return: []

Text to analyze:
{text}
"""
    )
