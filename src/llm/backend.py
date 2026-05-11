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
    return send_to_model(f"""
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
""")


def merge_prompt(text: str) -> str:
    return send_to_model(f"""
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
""")


def extract_user_functions_prompt(text: str) -> str:
    return send_to_model(f"""
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
""")


def extract_user_function_components(funct_reqs: str, ufs: str) -> str:
    # TODO: Finish prompt
    return send_to_model(f"""
You are an assistant that analyzes functional software requirements and their COCOMO II user functions, and extracts data elements, record elements, and referenced file types.

Task:
For each user function listed in the input text, extract all data elements, record elements, and referenced file types. Use the original requirements from which the user functions were derived as evidence.

User function types:
- ILF: Data that the system stores, uses, and manages internally.
- EIF: Data that is managed by an external system and used by the system, but not modified by it.
- EI: Input from outside the system that creates, modifies, or deletes internal data.
- EO: Output for which data is processed, combined, calculated, or derived before being presented.
- EQ: A request where an input directly leads to an output without modifying internal data and without additional processing or calculation.

Data Element:
A single user-recognizable data field in an input or output.

Example:
The user enters name, address, and date of birth. The system then displays the order number.

4 Data Elements:
- Name
- Address
- Date of birth
- Order number

Record Element:
A logically distinguishable subgroup within a stored object.

Example:
A database stores customers. A customer consists of master data, contact data, and contract data.

3 Record Elements:
- Master data
- Contact data
- Contract data

File Type Referenced:
A distinct data object that a function reads or modifies, regardless of whether it is internal (ILF) or external (EIF).

Example:
The system creates an order based on customer data and product data.

3 File Types Referenced:
- Customers
- Products
- Orders

Output format:
Return a valid JSON object containing the user functions and their extracted data elements, record elements, and referenced file types.
Your entire response must be valid JSON only. Do not include explanations or additional text.

The JSON object must follow this structure:

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

Rules:
- Only use information from the input requirements and the listed user functions.
- Do not invent, infer, generalize, or add new elements.
- Only extract elements that are explicitly stated or clearly identifiable in the requirements.
- Do not extract technical implementation details unless they are user-recognizable business data.
- Do not count the same element multiple times within the same user function.
- Keep separate user functions separate.
- Do not merge elements from different user functions.
- Use short and clear names.
- Preserve the input language where possible.
- If no elements can be identified for a field, return an empty array.
- If a category has no user functions, return an empty array for that category.
- Always return all five keys: "ILF", "EIF", "EI", "EO", "EQ".
- Do not add any keys other than "ILF", "EIF", "EI", "EO", "EQ".
- Your entire response must be valid JSON only.

Classification rules:
- For ILF and EIF, extract RET and DET.
- For EI, EO, and EQ, extract FTR and DET.
- RET means Record Element Type: a logically distinguishable subgroup within an ILF or EIF.
- DET means Data Element Type: a unique user-recognizable data field.
- FTR means File Type Referenced: a distinct ILF or EIF read or modified by an EI, EO, or EQ.
- For ILF, DET contains the user-recognizable data fields stored in the internal data group.
- For EIF, DET contains the user-recognizable data fields read from the external data group.
- For EI, DET contains the input fields or control information provided by the user or external actor.
- For EO, DET contains the output fields, calculated values, derived values, messages, or control information shown to the user or external actor.
- For EQ, DET contains the input fields used for the request and the output fields directly retrieved and displayed.
- For EI, FTR includes the internal data groups created, modified, or deleted, and any data groups read during the input process.
- For EO, FTR includes the data groups read to generate, calculate, derive, aggregate, or format the output.
- For EQ, FTR includes the data groups read to retrieve and display existing data.
- Do not add FTR to ILF or EIF.
- Do not add RET to EI, EO, or EQ.

Extraction example:

Requirement:
The system stores customer data, including master data, contact data, and contract data. Master data includes customer number, name, and date of birth. Contact data includes address, email address, and phone number. Contract data includes contract number, contract type, and contract status. Users can create a customer by entering name, date of birth, address, email address, and phone number. The system can display customer details based on a customer number.

User functions:
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
    }}
  ],
  "EO": [],
  "EQ": [
    {{
      "description": "Display customer details"
    }}
  ]
}}

Output:
{{
  "ILF": [
    {{
      "description": "Customers",
      "RET": [
        "Master data",
        "Contact data",
        "Contract data"
      ],
      "DET": [
        "Customer number",
        "Name",
        "Date of birth",
        "Address",
        "Email address",
        "Phone number",
        "Contract number",
        "Contract type",
        "Contract status"
      ]
    }}
  ],
  "EIF": [],
  "EI": [
    {{
      "description": "Create customer",
      "FTR": [
        "Customers"
      ],
      "DET": [
        "Name",
        "Date of birth",
        "Address",
        "Email address",
        "Phone number"
      ]
    }}
  ],
  "EO": [],
  "EQ": [
    {{
      "description": "Display customer details",
      "FTR": [
        "Customers"
      ],
      "DET": [
        "Customer number",
        "Customer details"
      ]
    }}
  ]
}}

Input:
User functions:
{ufs}

Original requirements:
{funct_reqs}
""")
