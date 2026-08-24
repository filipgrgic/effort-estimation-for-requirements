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


def estimate_prec_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the four component ratings used to determine
the COCOMO II scale driver Precedentedness (PREC).

Definition:
PREC describes how familiar and precedented the planned software project is.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may not contain information about the
organization, development team, or previous projects.

A criterion must only be rated when the input contains sufficient evidence.
If the input does not provide sufficient information for a criterion, return
null for that criterion.

Task:
Assess the following four PREC criteria independently:

1. Product objectives understanding:
   Assess the organization's understanding of the objectives of the planned
   product.

   Only assign a rating when the input provides information about the
   organization's or stakeholders' understanding of the product objectives.

   According to the COCOMO II rating guidance:
   - General understanding corresponds to Very Low precedentedness.
   - Considerable understanding corresponds to the Nominal / High range.
   - Thorough understanding corresponds to Extra High precedentedness.

   Do not infer this criterion from:
   - the clarity or completeness of the requirements
   - the commonness of the application domain
   - the amount of functionality described

2. Related software experience:
   Assess the organization's or development team's experience in working with
   related software systems.

   Only assign a rating when the input provides information about experience
   with related software systems.

   According to the COCOMO II rating guidance:
   - Moderate experience corresponds to Very Low precedentedness.
   - Considerable experience corresponds to the Nominal / High range.
   - Extensive experience corresponds to Extra High precedentedness.

   Do not infer experience merely because the planned system uses common
   technologies or implements common functionality.

3. Concurrent hardware and operations:
   Assess the extent to which associated new hardware and operational
   procedures must be developed concurrently with the software.

   This criterion may be assessed when the input provides information about
   concurrent development of associated new hardware or operational
   procedures.

   According to the COCOMO II rating guidance:
   - Extensive concurrent development corresponds to Very Low precedentedness.
   - Moderate concurrent development corresponds to the Nominal / High range.
   - Some concurrent development corresponds to Extra High precedentedness.

   Do not assume that no new hardware or operational procedures are required
   merely because they are not mentioned.

4. Architecture and algorithm innovation:
   Assess the need for innovative data-processing architectures or algorithms.

   This criterion may be assessed when the input provides information about
   the need for innovative data-processing architectures or algorithms.

   According to the COCOMO II rating guidance:
   - Considerable need for innovation corresponds to Very Low precedentedness.
   - Some need for innovation corresponds to the Nominal / High range.
   - Minimal need for innovation corresponds to Extra High precedentedness.

   Do not treat ordinary technical complexity, performance requirements,
   integrations, security requirements, or custom functionality as innovation
   by themselves.

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

Rules:
- Only use information from the provided requirements.
- Do not invent organizational knowledge, experience, technologies, hardware,
  procedures, architectures, or algorithms.
- Use null when a criterion cannot be assessed reliably.
- The absence of information is not evidence of a favorable or unfavorable
  condition.
- Do not assume that no new hardware or operational procedures are required
  merely because they are not mentioned.
- Do not infer related software experience merely because the requirements
  describe a common type of system.
- Do not use the clarity or completeness of the requirements as evidence of the
  organization's understanding of the product objectives.
- Do not treat ordinary technical complexity as architectural or algorithmic
  innovation.
- A requirement is innovative only when it clearly calls for a novel,
  uncommon, experimental, or technically unprecedented architecture or
  algorithm.
- Evaluate every criterion independently.
- Each rating must be an integer from 1 to 6 or null.
- Do not calculate or return the final PREC rating.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly these four keys:
- "product_objectives_understanding_rating"
- "related_software_experience_rating"
- "concurrent_hardware_and_operations_rating"
- "architecture_and_algorithm_innovation_rating"

Each value must be either:
- an integer from 1 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "product_objectives_understanding_rating": 3,
  "related_software_experience_rating": null,
  "concurrent_hardware_and_operations_rating": 4,
  "architecture_and_algorithm_innovation_rating": 2
}}

Requirements:
{text}
""")


def estimate_flex_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the three component ratings used to determine
the COCOMO II scale driver Development Flexibility (FLEX).

Definition:
FLEX describes the degree of flexibility available to the software project,
ranging from rigorous constraints to general goals.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may not contain information about project
constraints, external interface specifications, or schedule priorities.

A criterion must only be rated when the input contains sufficient evidence.
If the input does not provide sufficient information for a criterion, return
null for that criterion.

Task:
Assess the following three FLEX criteria independently:

1. Conformance with pre-established requirements:
   Assess the need for the software to conform with pre-established
   requirements.

   Only assign a rating when the input provides information about the degree to
   which pre-established requirements constrain the software.

   According to the COCOMO II rating guidance:
   - Full conformance corresponds to Very Low flexibility.
   - Considerable conformance corresponds to the Nominal / High range.
   - Basic conformance corresponds to Extra High flexibility.

   Do not assume that requirements are flexible or rigid merely because they
   are described in detail.

2. Conformance with external interface specifications:
   Assess the need for the software to conform with external interface
   specifications.

   Only assign a rating when the input provides information about the degree to
   which external interface specifications constrain the software.

   According to the COCOMO II rating guidance:
   - Full conformance corresponds to Very Low flexibility.
   - Considerable conformance corresponds to the Nominal / High range.
   - Basic conformance corresponds to Extra High flexibility.

   Do not assume a particular degree of conformance merely because an external
   interface or external system is mentioned.

3. Premium on early completion:
   Assess the importance placed on completing the software project early.

   Only assign a rating when the input explicitly provides information about
   the importance or priority of early completion.

   According to the COCOMO II rating guidance:
   - A high premium on early completion corresponds to Very Low flexibility.
   - A medium premium corresponds to the Nominal / High range.
   - A low premium corresponds to Extra High flexibility.

   Do not infer schedule pressure or a premium on early completion merely from
   the scope, complexity, or importance of the software.

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

Rules:
- Only use information from the provided requirements.
- Do not invent project constraints, interface specifications, schedule
  priorities, or flexibility.
- Use null when a criterion cannot be assessed reliably.
- The absence of information is not evidence of a favorable or unfavorable
  condition.
- Do not assume that pre-established requirements are flexible or rigid merely
  because they are present in the requirements.
- Do not assume a particular degree of conformance merely because an external
  interface is mentioned.
- Do not infer a premium on early completion unless the requirements provide
  information about schedule priority or early completion.
- Evaluate every criterion independently.
- Each rating must be an integer from 1 to 6 or null.
- Do not calculate or return the final FLEX rating.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly these three keys:
- "preestablished_requirements_conformance_rating"
- "external_interface_conformance_rating"
- "early_completion_premium_rating"

Each value must be either:
- an integer from 1 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "preestablished_requirements_conformance_rating": 2,
  "external_interface_conformance_rating": null,
  "early_completion_premium_rating": 5
}}

Requirements:
{text}
""")


def estimate_resl_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the seven component ratings used to determine
the COCOMO II scale driver Architecture / Risk Resolution (RESL).

Definition:
RESL describes the extent to which the software architecture is established and
project risks are identified and resolved.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may not contain information about risk
management activities, project schedules, budgets, software architects, tools,
or architectural planning.

A criterion must only be rated when the input contains sufficient evidence.
If the input does not provide sufficient information for a criterion, return
null for that criterion.

Task:
Assess the following seven RESL criteria independently:

1. Risk management plan:
   Assess the extent to which the Risk Management Plan identifies critical risk
   items and establishes milestones for resolving them by the Product Design
   Review (PDR).

   Only assign a rating when the input provides information about a Risk
   Management Plan, identified critical risks, or planned milestones for
   resolving those risks.

   According to the COCOMO II rating guidance:
   - None corresponds to Very Low risk resolution.
   - Little corresponds to Low risk resolution.
   - Some corresponds to Nominal risk resolution.
   - Generally corresponds to High risk resolution.
   - Mostly corresponds to Very High risk resolution.
   - Fully corresponds to Extra High risk resolution.

   Do not infer the existence or quality of a Risk Management Plan merely
   because technical risks or difficult requirements are mentioned.

2. Schedule, budget, and milestones compatibility:
   Assess the extent to which the schedule, budget, and internal milestones
   through PDR are compatible with the Risk Management Plan.

   Only assign a rating when the input provides information about both project
   planning constraints and their compatibility with risk management
   activities.

   According to the COCOMO II rating guidance:
   - None corresponds to Very Low risk resolution.
   - Little corresponds to Low risk resolution.
   - Some corresponds to Nominal risk resolution.
   - Generally corresponds to High risk resolution.
   - Mostly corresponds to Very High risk resolution.
   - Fully corresponds to Extra High risk resolution.

   Do not infer compatibility from the existence of a schedule, budget,
   milestones, or risk management information alone.

3. Development schedule devoted to architecture:
   Assess the percentage of the development schedule devoted to establishing
   the software architecture, given the general product objectives.

   Only assign a rating when the input provides sufficient information about
   the amount or percentage of the development schedule allocated to
   establishing the architecture.

   According to the COCOMO II rating guidance:
   - About 5% corresponds to Very Low risk resolution.
   - About 10% corresponds to Low risk resolution.
   - About 17% corresponds to Nominal risk resolution.
   - About 25% corresponds to High risk resolution.
   - About 33% corresponds to Very High risk resolution.
   - About 40% corresponds to Extra High risk resolution.

   Do not infer the amount of architecture work from the architectural
   complexity, amount of functionality, or level of detail in the
   requirements.

4. Availability of top software architects:
   Assess the percentage of required top software architects available to the
   project.

   Only assign a rating when the input provides information about both the
   required and available top software architecture capability.

   According to the COCOMO II rating guidance:
   - About 20% availability corresponds to Very Low risk resolution.
   - About 40% availability corresponds to Low risk resolution.
   - About 60% availability corresponds to Nominal risk resolution.
   - About 80% availability corresponds to High risk resolution.
   - About 100% availability corresponds to Very High risk resolution.
   - About 120% availability corresponds to Extra High risk resolution.

   Do not infer architect availability from the technologies, architecture,
   team size, or complexity described in the requirements.

5. Tool support for risk resolution and architecture:
   Assess the tool support available for resolving risk items and for
   developing and verifying architectural specifications.

   Only assign a rating when the input provides information about tools used
   for risk resolution or for developing or verifying architectural
   specifications.

   According to the COCOMO II rating guidance:
   - No tool support corresponds to Very Low risk resolution.
   - Little tool support corresponds to Low risk resolution.
   - Some tool support corresponds to Nominal risk resolution.
   - Good tool support corresponds to High risk resolution.
   - Strong tool support corresponds to Very High risk resolution.
   - Full tool support corresponds to Extra High risk resolution.

   Do not infer tool support merely because programming languages,
   frameworks, development tools, or technologies are mentioned.

6. Uncertainty in key architecture drivers:
   Assess the level of uncertainty in key architecture drivers such as the
   mission, user interface, COTS components, hardware, technology, and
   performance.

   Only assign a rating when the input provides evidence about the uncertainty
   or stability of these architecture drivers.

   According to the COCOMO II rating guidance:
   - Extreme uncertainty corresponds to Very Low risk resolution.
   - Significant uncertainty corresponds to Low risk resolution.
   - Considerable uncertainty corresponds to Nominal risk resolution.
   - Some uncertainty corresponds to High risk resolution.
   - Little uncertainty corresponds to Very High risk resolution.
   - Very little uncertainty corresponds to Extra High risk resolution.

   Do not treat the mere presence of requirements concerning user interfaces,
   COTS components, hardware, technologies, or performance as evidence of
   uncertainty.

7. Number and criticality of risk items:
   Assess the number and criticality of identified project risk items.

   Only assign a rating when the input provides sufficient information about
   both the number and criticality of identified risk items.

   According to the COCOMO II rating guidance:
   - More than 10 critical risk items corresponds to Very Low risk resolution.
   - 5 to 10 critical risk items corresponds to Low risk resolution.
   - 2 to 4 critical risk items corresponds to Nominal risk resolution.
   - 1 critical risk item corresponds to High risk resolution.
   - More than 5 non-critical risk items corresponds to Very High risk
     resolution.
   - Fewer than 5 non-critical risk items corresponds to Extra High risk
     resolution.

   Do not count requirements, features, integrations, constraints, or technical
   challenges as risk items unless the input identifies them as risks or
   otherwise clearly provides sufficient evidence that they are project risk
   items.

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

Rules:
- Only use information from the provided requirements.
- Do not invent risk management plans, schedules, budgets, milestones,
  architects, tools, architecture decisions, uncertainties, or risk items.
- Use null when a criterion cannot be assessed reliably.
- The absence of information is not evidence of a favorable or unfavorable
  condition.
- Do not assume that risks have been identified or resolved merely because they
  are not mentioned.
- Do not assume that an architecture has been established merely because
  architectural elements or technologies are described.
- Do not infer project management activities from product requirements.
- Do not infer architect availability from the existence or complexity of an
  architecture.
- Do not infer architecture tool support from ordinary development tools or
  technologies.
- Do not treat product requirements or technical complexity as identified risk
  items by themselves.
- Evaluate every criterion independently.
- Each rating must be an integer from 1 to 6 or null.
- Do not calculate or return the final RESL rating.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly these seven keys:
- "risk_management_plan_rating"
- "schedule_budget_milestones_compatibility_rating"
- "architecture_schedule_percentage_rating"
- "top_software_architects_availability_rating"
- "architecture_risk_tool_support_rating"
- "architecture_driver_uncertainty_rating"
- "risk_items_rating"

Each value must be either:
- an integer from 1 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "risk_management_plan_rating": null,
  "schedule_budget_milestones_compatibility_rating": null,
  "architecture_schedule_percentage_rating": 3,
  "top_software_architects_availability_rating": null,
  "architecture_risk_tool_support_rating": 4,
  "architecture_driver_uncertainty_rating": 2,
  "risk_items_rating": 3
}}

Requirements:
{text}
""")


def estimate_rely_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Required Software Reliability (RELY).

Definition:
RELY describes the extent to which the software must perform its intended
function over a period of time. The rating is determined by the consequences
of a software failure.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about the
consequences of software failures.

The criterion must only be rated when the input contains sufficient evidence.
If the input does not provide sufficient information about the consequences of
software failure, return null.

Task:
Assess the following RELY criterion:

1. Consequences of software failure:
   Assess the severity of the consequences if the software fails to perform its
   intended function.

   Only assign a rating when the input provides information about the
   consequences or losses caused by software failure.

   According to the COCOMO II rating guidance:
   - Slight inconvenience corresponds to Very Low reliability.
   - Low, easily recoverable losses correspond to Low reliability.
   - Moderate, easily recoverable losses correspond to Nominal reliability.
   - High financial loss corresponds to High reliability.
   - Risk to human life corresponds to Very High reliability.

   Do not infer failure consequences merely from:
   - the type or domain of the software
   - the importance of a feature or business process
   - security requirements
   - performance requirements
   - availability or uptime requirements
   - the amount or complexity of functionality

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- null: The criterion cannot be assessed from the requirements.

RELY does not have an Extra High rating in the COCOMO II rating scale.

Rules:
- Only use information from the provided requirements.
- Do not invent consequences, losses, safety implications, or failure
  scenarios.
- Use null when the consequences of software failure cannot be assessed
  reliably.
- The absence of information about failures is not evidence of low or high
  required reliability.
- Do not assume high reliability merely because the software is described as
  important, critical, secure, highly available, or performance-sensitive.
- Do not assume that software in a particular application domain has specific
  failure consequences unless those consequences are supported by the
  requirements.
- Base the rating on the consequences of failure, not on general software
  quality requirements.
- The rating must be an integer from 1 to 5 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "rely"

The value must be either:
- an integer from 1 to 5, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "rely": 3
}}

Requirements:
{text}
""")


def estimate_data_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Data Base Size (DATA).

Definition:
DATA describes the effect of large data requirements on software development.
The rating is determined by the ratio of database size in bytes to program size
in source lines of code (SLOC).

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about the expected
database size or program size.

The criterion must only be rated when the input contains sufficient evidence
to determine the ratio between database size and program size.
If the input does not provide sufficient information, return null.

Task:
Assess the following DATA criterion:

1. Database size relative to program size:
   Assess the ratio D/P, where:
   - D is the database size in bytes.
   - P is the program size in SLOC.

   Only assign a rating when the input provides sufficient quantitative
   information to determine both the database size and the program size and
   therefore calculate D/P.

   According to the COCOMO II rating guidance:
   - D/P < 10 corresponds to Low.
   - 10 <= D/P < 100 corresponds to Nominal.
   - 100 <= D/P < 1000 corresponds to High.
   - D/P >= 1000 corresponds to Very High.

   Do not infer database size or program size merely from:
   - the number of entities, tables, files, or data types
   - the amount or complexity of functionality
   - the presence of a database or database management system
   - descriptions such as small, large, or data-intensive without sufficient
     quantitative information
   - the number of users or expected transactions

Rating scale:
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- null: The criterion cannot be assessed from the requirements.

DATA does not have a Very Low or Extra High rating in the COCOMO II rating
scale.

Rules:
- Only use information from the provided requirements.
- Do not invent or estimate database size, program size, record sizes, numbers
  of records, or SLOC when they are not provided.
- Use null when D/P cannot be determined reliably.
- The absence of information about database size is not evidence of a small
  database.
- The presence of a database is not sufficient to determine DATA.
- Do not infer database size from the number of database tables, entities,
  files, or data structures alone.
- Do not infer program size from the amount or complexity of functionality
  described in the requirements.
- Base the rating on the D/P ratio defined by COCOMO II, not on a qualitative
  judgment of whether the application appears data-intensive.
- The rating must be an integer from 2 to 5 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "data"

The value must be either:
- an integer from 2 to 5, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "data": 4
}}

Requirements:
{text}
""")


def estimate_docu_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Documentation Match to Life-Cycle Needs (DOCU).

Definition:
DOCU describes the suitability of the project's documentation to its
life-cycle needs.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about required
documentation or documentation-related project needs.

The criterion must only be rated when the input contains sufficient evidence
about the relationship between the required documentation and the project's
life-cycle needs.
If the input does not provide sufficient information, return null.

Task:
Assess the following DOCU criterion:

1. Documentation match to life-cycle needs:
   Assess how well the required project documentation matches the software
   project's life-cycle needs.

   Only assign a rating when the input provides sufficient information about
   both the documentation required for the project and how that documentation
   relates to the project's life-cycle needs.

   According to the COCOMO II rating guidance:
   - Many life-cycle needs uncovered corresponds to Very Low.
   - Some life-cycle needs uncovered corresponds to Low.
   - Documentation right-sized to life-cycle needs corresponds to Nominal.
   - Documentation excessive for life-cycle needs corresponds to High.
   - Documentation very excessive for life-cycle needs corresponds to Very
     High.

   Do not infer the documentation rating merely from:
   - the amount or detail of the software requirements
   - the complexity or size of the software
   - the presence of individual documentation requirements
   - the importance or criticality of the software
   - the use of comments, README files, or technical descriptions unless their
     relationship to life-cycle needs is provided

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- null: The criterion cannot be assessed from the requirements.

DOCU does not have an Extra High rating in the COCOMO II rating scale.

Rules:
- Only use information from the provided requirements.
- Do not invent documentation requirements or life-cycle needs.
- Use null when the match between documentation and life-cycle needs cannot be
  assessed reliably.
- The absence of documentation information is not evidence that life-cycle
  needs are uncovered.
- The presence of documentation requirements alone is not sufficient to
  determine whether documentation is insufficient, right-sized, or excessive.
- Do not use the detail or completeness of the provided requirements as
  evidence for the amount or suitability of project documentation.
- Base the rating on the suitability of documentation to life-cycle needs, not
  on a general judgment of documentation quality.
- The rating must be an integer from 1 to 5 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "docu"

The value must be either:
- an integer from 1 to 5, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "docu": 3
}}

Requirements:
{text}
""")


def estimate_cplx_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the five component ratings used to determine
the COCOMO II cost driver Product Complexity (CPLX).

Definition:
CPLX describes the complexity of the software product or subsystem across
control operations, computational operations, device-dependent operations,
data management operations, and user interface management operations.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about the types and
complexity of operations the software must perform.

A criterion must only be rated when the input contains sufficient evidence.
If the input does not provide sufficient information for a criterion, return
null for that criterion.

Task:
Assess the following five CPLX criteria independently:

1. Control operations:
   Assess the complexity of control operations required by the software.

   Only assign a rating when the input provides sufficient information about
   control logic, intermodule control, distributed processing, real-time
   control, interrupt handling, task synchronization, resource scheduling, or
   comparable control operations.

   According to the COCOMO II rating guidance:
   - Straight-line control with few non-nested structured programming
     operators and simple module composition corresponds to Very Low.
   - Straightforward nesting of structured programming operators with mostly
     simple predicates corresponds to Low.
   - Mostly simple nesting with some intermodule control, decision tables,
     simple callbacks or message passing corresponds to Nominal.
   - Highly nested control with many compound predicates, queue or stack
     control, homogeneous distributed processing, or single-processor soft
     real-time control corresponds to High.
   - Reentrant or recursive control, fixed-priority interrupt handling, task
     synchronization, complex callbacks, heterogeneous distributed processing,
     or single-processor hard real-time control corresponds to Very High.
   - Multiple-resource scheduling with dynamically changing priorities,
     microcode-level control, or distributed hard real-time control
     corresponds to Extra High.

   Do not infer control complexity merely from:
   - the number of features or requirements
   - the overall size of the software
   - ordinary workflow or business rules without evidence of the corresponding
     control complexity
   - the presence of multiple software components by itself

2. Computational operations:
   Assess the complexity of computational operations required by the software.

   Only assign a rating when the input provides sufficient information about
   mathematical, statistical, numerical, matrix, vector, parallel, or other
   computational operations.

   According to the COCOMO II rating guidance:
   - Evaluation of simple expressions corresponds to Very Low.
   - Evaluation of moderate-level expressions corresponds to Low.
   - Use of standard mathematical or statistical routines and basic
     matrix/vector operations corresponds to Nominal.
   - Basic numerical analysis, such as multivariate interpolation or ordinary
     differential equations, with basic truncation or roundoff concerns
     corresponds to High.
   - Difficult but structured numerical analysis, such as near-singular matrix
     equations or partial differential equations, or simple parallelization
     corresponds to Very High.
   - Difficult and unstructured numerical analysis, highly accurate analysis
     of noisy or stochastic data, or complex parallelization corresponds to
     Extra High.

   Do not infer computational complexity merely from:
   - processing large amounts of data
   - ordinary arithmetic or business calculations
   - performance requirements
   - the use of algorithms without information about their computational
     characteristics

3. Device-dependent operations:
   Assess the complexity of device-dependent operations required by the
   software.

   Only assign a rating when the input provides sufficient information about
   input/output processing, hardware or device interaction, interrupt handling,
   communication-line handling, embedded operation, or other device-dependent
   behavior.

   According to the COCOMO II rating guidance:
   - Simple read and write statements with simple formats correspond to Very
     Low.
   - I/O requiring no knowledge of particular processor or device
     characteristics and performed at the GET/PUT level corresponds to Low.
   - I/O processing involving device selection, status checking, and error
     processing corresponds to Nominal.
   - Operations at the physical I/O level, including physical storage address
     translations or optimized I/O overlap, correspond to High.
   - Interrupt diagnosis, servicing or masking, communication-line handling,
     or performance-intensive embedded systems correspond to Very High.
   - Device timing-dependent coding, micro-programmed operations, or
     performance-critical embedded systems correspond to Extra High.

   Do not infer device-dependent complexity merely from:
   - deployment on a particular hardware platform
   - the use of common peripherals
   - communication with external systems
   - the presence of hardware requirements without information about the
     required device-level operations

4. Data management operations:
   Assess the complexity of data management operations required by the
   software.

   Only assign a rating when the input provides sufficient information about
   data structures, files, database operations, data restructuring, triggers,
   distributed databases, search optimization, or comparable data management
   operations.

   According to the COCOMO II rating guidance:
   - Simple arrays in main memory or simple COTS database queries and updates
     correspond to Very Low.
   - Single-file subsetting without data structure changes, edits, or
     intermediate files, or moderately complex COTS database queries and
     updates corresponds to Low.
   - Multi-file input with single-file output, simple structural changes or
     edits, or complex COTS database queries and updates corresponds to
     Nominal.
   - Simple triggers activated by data stream contents or complex data
     restructuring corresponds to High.
   - Distributed database coordination, complex triggers, or search
     optimization corresponds to Very High.
   - Highly coupled dynamic relational or object structures, or natural
     language data management, corresponds to Extra High.

   Do not infer data management complexity merely from:
   - the presence of a database
   - the number of entities or database tables
   - the amount of stored data
   - CRUD functionality by itself
   - the number of users or transactions

5. User interface management operations:
   Assess the complexity of user interface management operations required by
   the software.

   Only assign a rating when the input provides sufficient information about
   forms, graphical user interfaces, widget sets, voice interfaces, multimedia,
   dynamic graphics, virtual reality, or comparable user interface operations.

   According to the COCOMO II rating guidance:
   - Simple input forms and report generators correspond to Very Low.
   - Use of simple graphical user interface builders corresponds to Low.
   - Simple use of a widget set corresponds to Nominal.
   - Widget set development or extension, simple voice I/O, or multimedia
     corresponds to High.
   - Moderately complex 2D or 3D dynamic graphics or multimedia corresponds to
     Very High.
   - Complex multimedia or virtual reality corresponds to Extra High.

   Do not infer user interface complexity merely from:
   - the presence of a graphical user interface
   - the number of screens or pages
   - visual design or styling requirements
   - responsive design by itself
   - the number of user roles

Rating scale:
- 1: Very Low
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

Rules:
- Only use information from the provided requirements.
- Do not invent control logic, computational operations, device interactions,
  data management behavior, or user interface behavior.
- Use null when a criterion cannot be assessed reliably.
- The absence of information is not evidence of low complexity.
- Only rate complexity areas that are supported by the requirements.
- Do not rate an area merely because it is common for the described type of
  software.
- Do not infer implementation-level complexity from the number of features,
  requirements, components, users, or integrations.
- Distinguish computational complexity from data management complexity.
- Distinguish device-dependent operations from ordinary interaction with
  external software systems.
- Distinguish user interface complexity from visual appearance or the number
  of screens.
- Evaluate every criterion independently.
- Each rating must be an integer from 1 to 6 or null.
- Do not calculate or return the final CPLX rating.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly these five keys:
- "control_operations_rating"
- "computational_operations_rating"
- "device_dependent_operations_rating"
- "data_management_operations_rating"
- "user_interface_management_operations_rating"

Each value must be either:
- an integer from 1 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "control_operations_rating": 3,
  "computational_operations_rating": null,
  "device_dependent_operations_rating": null,
  "data_management_operations_rating": 4,
  "user_interface_management_operations_rating": 2
}}

Requirements:
{text}
""")


def estimate_ruse_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Required Reusability (RUSE).

Definition:
RUSE describes the additional effort required to construct software components
that are intended for reuse in the current or future projects. This additional
effort results from creating more generic designs, more elaborate
documentation, and more extensive testing so that components are suitable for
reuse in other applications.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about whether
components must be designed for reuse and the scope across which they are
intended to be reused.

The criterion must only be rated when the input contains sufficient evidence
about the required scope of reuse.
If the input does not provide sufficient information, return null.

Task:
Assess the following RUSE criterion:

1. Required scope of reuse:
   Assess the scope across which the software or its components are required to
   be reusable.

   Only assign a rating when the input provides sufficient information about
   whether reuse is required and the scope across which that reuse is intended.

   According to the COCOMO II rating guidance:
   - No required reuse corresponds to Low.
   - Reuse across the project corresponds to Nominal.
   - Reuse across the program corresponds to High.
   - Reuse across the product line corresponds to Very High.
   - Reuse across multiple product lines corresponds to Extra High.

   Do not infer required reusability merely from:
   - modular or component-based architecture
   - object-oriented design
   - the use of libraries, frameworks, or shared utilities
   - the use of existing reusable components or COTS software
   - code duplication avoidance
   - maintainability or extensibility requirements
   - general statements about software quality

Rating scale:
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

RUSE does not have a Very Low rating in the COCOMO II rating scale.

Rules:
- Only use information from the provided requirements.
- Do not invent reuse requirements or a reuse scope.
- Use null when the required scope of reuse cannot be assessed reliably.
- The absence of a reuse requirement is not evidence that no reuse is
  required.
- Assign Low only when the input provides evidence that no reuse is required.
- Do not confuse required reusability of newly developed software with the use
  of existing reusable software, libraries, frameworks, or COTS components.
- Do not assume that modular, generic, maintainable, or extensible software is
  intended for reuse.
- A general statement that software should be reusable is not sufficient to
  determine a rating unless the intended scope of reuse can be identified.
- Base the rating on the required reuse scope defined by COCOMO II.
- The rating must be an integer from 2 to 6 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "ruse"

The value must be either:
- an integer from 2 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "ruse": 4
}}

Requirements:
{text}
""")


def estimate_time_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Execution Time Constraint (TIME).

Definition:
TIME describes the execution time constraint imposed on the software system or
subsystem. The rating is determined by the percentage of the available
execution time resource expected to be consumed by the system or subsystem.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about execution time
resource usage or constraints.

The criterion must only be rated when the input contains sufficient evidence
about the percentage of available execution time expected to be used.
If the input does not provide sufficient information, return null.

Task:
Assess the following TIME criterion:

1. Use of available execution time:
   Assess the percentage of the available execution time resource expected to
   be consumed by the software system or subsystem.

   Only assign a rating when the input provides sufficient information about
   the amount or percentage of the available execution time resource expected
   to be used.

   According to the COCOMO II rating guidance:
   - 50% or less use of available execution time corresponds to Nominal.
   - About 70% use of available execution time corresponds to High.
   - About 85% use of available execution time corresponds to Very High.
   - About 95% use of available execution time corresponds to Extra High.

   Do not infer execution time resource usage merely from:
   - response time requirements
   - latency requirements
   - throughput requirements
   - general performance requirements
   - real-time functionality
   - the amount or complexity of computation
   - the number of users or transactions
   - statements that the software must be fast or efficient

Rating scale:
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

TIME does not have a Very Low or Low rating in the COCOMO II rating scale.

Rules:
- Only use information from the provided requirements.
- Do not invent execution time resource limits, available execution time, or
  expected execution time consumption.
- Use null when the percentage of available execution time expected to be used
  cannot be assessed reliably.
- The absence of an execution time constraint is not evidence of a Nominal
  rating.
- Do not infer execution time resource consumption from response time, latency,
  throughput, or other performance requirements alone.
- Do not infer a TIME rating merely because the software is described as
  real-time, performance-critical, computationally intensive, fast, or
  efficient.
- Base the rating on the percentage of available execution time resource
  consumed, as defined by COCOMO II.
- The rating must be an integer from 3 to 6 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "time"

The value must be either:
- an integer from 3 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "time": 4
}}

Requirements:
{text}
""")


def estimate_stor_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Main Storage Constraint (STOR).

Definition:
STOR describes the degree of main storage constraint imposed on the software
system or subsystem. The rating is determined by the percentage of available
main storage expected to be used by the system or subsystem.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about memory usage or
main storage constraints.

The criterion must only be rated when the input contains sufficient evidence
about the percentage of available main storage expected to be used.
If the input does not provide sufficient information, return null.

Task:
Assess the following STOR criterion:

1. Use of available main storage:
   Assess the percentage of the available main storage resource expected to be
   consumed by the software system or subsystem.

   Only assign a rating when the input provides sufficient information about
   the amount or percentage of available main storage expected to be used.

   According to the COCOMO II rating guidance:
   - 50% or less use of available storage corresponds to Nominal.
   - About 70% use of available storage corresponds to High.
   - About 85% use of available storage corresponds to Very High.
   - About 95% use of available storage corresponds to Extra High.

   Do not infer main storage usage merely from:
   - the amount of data processed or stored
   - database size
   - file sizes
   - the number of users or transactions
   - caching requirements
   - general performance requirements
   - the complexity or size of the software
   - statements that the software must use memory efficiently

Rating scale:
- 3: Nominal
- 4: High
- 5: Very High
- 6: Extra High
- null: The criterion cannot be assessed from the requirements.

STOR does not have a Very Low or Low rating in the COCOMO II rating scale.

Rules:
- Only use information from the provided requirements.
- Do not invent available storage capacity, expected memory consumption, or
  storage constraints.
- Use null when the percentage of available main storage expected to be used
  cannot be assessed reliably.
- The absence of a main storage constraint is not evidence of a Nominal rating.
- Do not infer main storage consumption from database size, stored data volume,
  file sizes, or the amount of functionality alone.
- Do not infer a STOR rating merely because the software is described as
  memory-intensive, performance-critical, large, or resource-efficient.
- Distinguish main storage usage from persistent database or file storage.
- Base the rating on the percentage of available main storage consumed, as
  defined by COCOMO II.
- The rating must be an integer from 3 to 6 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "stor"

The value must be either:
- an integer from 3 to 6, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "stor": 5
}}

Requirements:
{text}
""")


def estimate_pvol_prompt(text: str) -> str:
    return send_to_model(f"""
You are an assistant that assesses the COCOMO II cost driver
Platform Volatility (PVOL).

Definition:
PVOL describes the volatility of the platform on which the software product
depends. The platform includes the hardware and software such as operating
systems, database management systems, networks, and other infrastructure
software that the product calls on to perform its tasks. It also includes
compilers or assemblers supporting the development of the software system.

Available information:
You receive initial software requirements. These requirements primarily
describe the software product and may contain information about expected
changes to the hardware or software platform.

The criterion must only be rated when the input contains sufficient evidence
about the frequency of major or minor platform changes.
If the input does not provide sufficient information, return null.

Task:
Assess the following PVOL criterion:

1. Frequency of platform changes:
   Assess how frequently major or minor changes to the platform are expected
   to occur.

   Only assign a rating when the input provides sufficient information about
   the expected frequency of major or minor changes to the platform.

   According to the COCOMO II rating guidance:
   - Major changes every 12 months and minor changes every 1 month correspond
     to Low.
   - Major changes every 6 months and minor changes every 2 weeks correspond
     to Nominal.
   - Major changes every 2 months and minor changes every 1 week correspond
     to High.
   - Major changes every 2 weeks and minor changes every 2 days correspond
     to Very High.

   Do not infer platform volatility merely from:
   - the use of modern or rapidly evolving technologies
   - the use of third-party libraries, frameworks, or services
   - the number of technologies in the software stack
   - the complexity of the target platform
   - dependencies on external systems
   - planned software updates or releases that are not changes to the platform

Rating scale:
- 2: Low
- 3: Nominal
- 4: High
- 5: Very High
- null: The criterion cannot be assessed from the requirements.

PVOL does not have a Very Low or Extra High rating in the COCOMO II rating
scale.

Rules:
- Only use information from the provided requirements.
- Do not invent platform changes or their frequency.
- Use null when the frequency of platform changes cannot be assessed reliably.
- The absence of information about platform changes is not evidence of a
  stable platform.
- Do not assume a platform is stable merely because no changes are mentioned.
- Do not assume a platform is volatile merely because it uses modern,
  third-party, cloud-based, or frequently updated technologies.
- Distinguish changes to the software product itself from changes to the
  platform on which the product depends.
- Base the rating on the frequency of major or minor platform changes as
  defined by COCOMO II.
- The rating must be an integer from 2 to 5 or null.
- Do not add explanations, evidence, or additional keys.

Output format:
Return a valid JSON object only.
Do not include Markdown, comments, explanations, or additional text.

The JSON object must contain exactly this one key:
- "pvol"

The value must be either:
- an integer from 2 to 5, or
- null if the criterion cannot be assessed reliably.

Do not add any other keys.

Example output:

{{
  "pvol": 3
}}

Requirements:
{text}
""")
