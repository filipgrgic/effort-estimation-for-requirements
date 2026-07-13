from config import (
    MODEL_PATH,
    CONTEXT_SIZE,
    THREADS,
    GPU_LAYERS,
    TEMPERATURE,
    MAX_TOKENS,
)
from llama_cpp import Llama

_model = None


def load_model() -> Llama:
    global _model

    if _model is not None:
        return _model

    try:
        _model = Llama(
            model_path=MODEL_PATH,
            n_ctx=CONTEXT_SIZE,
            n_threads=THREADS,
            n_gpu_layers=GPU_LAYERS,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError("Could not load model") from e

    return _model


def send_to_model(prompt: str) -> str:
    llm = load_model()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": f"{prompt.rstrip()}\n\n/no_think"}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    cleaned_response = response["choices"][0]["message"]["content"]
    cleaned_response = cleaned_response.split("</think>", 1)[
        -1
    ].strip()  # Remove the <think>-block in the response

    return cleaned_response


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


def extract_user_function_components_prompt(funct_reqs: str, ufs: str) -> str:
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


def estimate_breakage_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that estimates the COCOMO II breakage factor, BRAK,
from initial software requirements.

Definition:
BRAK is the estimated percentage of code that will be developed but later
discarded because requirements evolve or change during development.

A BRAK value of 10 means that for every 100 units of delivered code,
an additional 10 units of code are expected to be developed and discarded
because of requirements changes.

Available information:
You receive only the initial requirements.

You do not know whether the requirements will actually change.
Therefore, estimate the likely breakage conservatively by analyzing indicators
in the requirements that could make later changes or redesign more likely.

Possible indicators include:
- ambiguous, vague, or incomplete requirements
- conflicting or inconsistent requirements
- undefined business rules
- unresolved design or technology decisions
- unclear system boundaries or responsibilities
- external integrations whose behavior is not sufficiently specified
- placeholders or provisional statements
- requirements that are likely to affect many parts of the system if changed
- future extensions that may require substantial changes to the initial
  architecture

Task:
Estimate one project-wide BRAK percentage for the requirements below.

Rules:
- Estimate only code likely to be discarded because requirements may evolve,
  change, become obsolete, or require substantial redesign during development.
- Do not include additional effort that does not result from changing
  requirements.
- Do not include effort caused only by complexity, defects, testing,
  documentation, low productivity, schedule pressure, maintenance,
  reuse, or re-engineering.
- Do not increase BRAK merely because the project is large or technically
  difficult.
- Performance, security, privacy, deployment, and reliability requirements
  do not by themselves imply high breakage.
- Configurable functionality does not by itself imply changing requirements.
- Requirements explicitly described only as future extensions should normally
  have little or no influence on BRAK unless they are likely to require major
  architectural changes during the initial development.
- Do not assume stakeholder instability, organizational problems, or changing
  business processes unless the requirements contain indicators for them.
- Treat ambiguous wording as a risk indicator, but not as proof that code will
  be discarded.
- Do not pretend to know that future changes will occur.
- When evidence is limited, prefer a low and conservative estimate.
- Avoid unsupported high estimates.
- Use the complete set of requirements to produce one overall value.

Estimation guidance:
- 0 to 3 percent:
  Requirements appear explicit, consistent, stable, and clearly scoped.

- More than 3 to 8 percent:
  Requirements appear mostly stable, with minor ambiguity or a small number
  of unresolved details.

- More than 8 to 15 percent:
  Several requirements are unclear, incomplete, provisional, or dependent
  on decisions that may cause moderate changes.

- More than 15 to 30 percent:
  There are significant unresolved requirements, conflicting statements,
  unstable external dependencies, or a substantial risk of redesign.

- More than 30 to 50 percent:
  There is strong evidence that major parts of the implementation may need
  to be changed or rebuilt.

- More than 50 percent:
  Exceptional. Use only when the initial requirements strongly indicate that
  large parts of the implementation are provisional or likely to be replaced.

Output format:
Return a valid JSON object only.
Do not include Markdown, explanations, comments, or additional text.

The JSON object must follow this exact structure:

{{
  "breakage": number
}}

Output rules:
- Do not add any other keys.
- "breakage" must be a JSON number, not a string.
- "breakage" must be between 0 and 100.
- Use at most one decimal place.
- Values above 30 require strong evidence in the requirements.
- Values above 50 require exceptional evidence in the requirements.

Requirements:
{text}                   
    """)
