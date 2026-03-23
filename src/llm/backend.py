import ollama
from config import MODEL_NAME


def send_to_model(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    except Exception as e:
        print(f"Error while calling model: {type(e).__name__}: {e}")
        raise


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


def merge_prompt(text: str) -> str:
    return send_to_model(
        f"""
You are an assistant that merges software requirements.

Task:
Merge duplicate requirements and requirements that have the same meaning from the JSON array below.

Output format:
Return a valid JSON array containing the merged requirements.
Your entire response must be valid JSON only. Do not include explanations or additional text.

Each requirement must follow this structure:

[
  {{
    "description": string,
    "type": "functional" | "non_functional" | "constraint"
  }}
]

Rules:
- Only use requirements from the input JSON array.
- Do not invent, infer, generalize, or add new requirements.
- Merge requirements only if they are duplicates or clearly express the same meaning.
- Do not merge requirements that are only loosely related or partially overlapping.
- Each merged requirement must be a separate JSON object.
- Preserve the original meaning as closely as possible.
- Prefer the clearest and most complete wording from the input when merging.
- Do not combine unrelated details into a broader requirement.
- The "type" must be exactly one of: "functional", "non_functional", "constraint".
- Only merge requirements with the same type.
- If two similar requirements have different types, keep them as separate requirements.
- Remove exact duplicates.
- Keep all unique requirements.
- Preserve the input language.
- Return requirements as a JSON array only.
- If the input array is empty, return: []

Input JSON:
{text}
"""
    )
