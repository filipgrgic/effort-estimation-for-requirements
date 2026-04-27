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


def extract_user_functions_prompt(text: str) -> str:
    return send_to_model(
        f"""
You are an assistant that analyzes functional software requirements and extracts COCOMO II user function types.

Task:
Extract all user functions from the requirements in the input text.

User function types:
- ILF: Data that the system stores, uses, and manages itself.
- EIF: Data that is managed by an external system and used by the system, but not modified by it.
- EI: Input from outside the system that creates, modifies, or deletes internal data.
- EO: Output where data is processed, combined, calculated, or derived before it is presented.
- EQ: Request where an input directly leads to an output without modifying internal data and without additional processing or calculation.

Output format:
Return a valid JSON object containing the extracted user functions.
Your entire response must be valid JSON only. Do not include explanations or additional text.

The JSON object must follow this structure:

{{
  "ILF": [
    {{
      "description": string
    }}
  ],
  "EIF": [
    {{
      "description": string
    }}
  ],
  "EI": [
    {{
      "description": string
    }}
  ],
  "EO": [
    {{
      "description": string
    }}
  ],
  "EQ": [
    {{
      "description": string
    }}
  ]
}}

Rules:
- Only use information from the input requirements.
- Do not invent, infer, generalize, or add new user functions.
- Only analyze functional requirements.
- Ignore non-functional requirements and constraints.
- Extract each distinct user function only once.
- Do not count the same function multiple times if it appears in several requirements.
- Keep separate functions separate.
- Do not merge different actions into one user function.
- Use short and clear descriptions.
- Preserve the input language where possible.
- If a category has no user functions, return an empty array for that category.
- Always return all five keys: "ILF", "EIF", "EI", "EO", "EQ".
- Do not add any keys other than "ILF", "EIF", "EI", "EO", "EQ".

Classification rules:
- Use ILF for internal data groups managed by the system, such as customers, orders, invoices, users, or bookings.
- Use EIF for external data groups that are only read or referenced by the system.
- Use EI for user or external actions that create, update, or delete internal data.
- Use EO for generated reports, invoices, summaries, exports, calculations, notifications, or derived outputs.
- Use EQ for searches, lookups, views, or retrievals that only display existing data without changing it.
- If a function both modifies data and shows a result, classify the modifying action as EI.
- If an output contains calculations, aggregations, formatting, or derived values, classify it as EO.
- If an output only retrieves and displays existing data, classify it as EQ.

Examples:

Requirement:
The system stores customer data and allows users to create and update customers.

Output:
{{
  "ILF": [
    {{
      "description": "Customers"
    }}
  ],
  "EIF": [],
  "EI": [
    {{
      "description": "Create customer"
    }},
    {{
      "description": "Update customer"
    }}
  ],
  "EO": [],
  "EQ": []
}}

Requirement:
The system reads product data from an external product catalog and allows users to search for products.

Output:
{{
  "ILF": [],
  "EIF": [
    {{
      "description": "Products"
    }}
  ],
  "EI": [],
  "EO": [],
  "EQ": [
    {{
      "description": "Search products"
    }}
  ]
}}

Requirement:
The user searches for orders and views order details. The system also generates a report with total revenue.

Output:
{{
  "ILF": [],
  "EIF": [],
  "EI": [],
  "EO": [
    {{
      "description": "Generate revenue report"
    }}
  ],
  "EQ": [
    {{
      "description": "Search orders"
    }},
    {{
      "description": "View order details"
    }}
  ]
}}

Input text:
{text}
"""
    )


def determine_complexity_prompt(funct_reqs: str, ufs: str) -> str:
    # TODO: Finish prompt
    return send_to_model(
        f"""

Output format:
{{
  "ILF": [
    {{
      "description": string,
      "RET": [string],
      "DET": [string]
    }}
  ],
  "EIF": [
    {{
      "description": string,
      "RET": [string],
      "DET": [string]
    }}
  ],
  "EI": [
    {{
      "description": string,
      "FTR": [string],
      "DET": [string]
    }}
  ],
  "EO": [
    {{
      "description": string,
      "FTR": [string],
      "DET": [string]
    }}
  ],
  "EQ": [
    {{
      "description": string,
      "FTR": [string],
      "DET": [string]
    }}
  ]
}}
"""
    )
