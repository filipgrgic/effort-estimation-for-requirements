from llm.backend import extract_user_functions_prompt
from llm.backend import extract_user_function_components_prompt
from schema.models import Requirement, RequirementType
import json


def estimate_size(reqs: list[Requirement]) -> float:
    funct_reqs = ""
    for req in reqs:
        # Only functional requirements are relevant for function points
        if req.type == RequirementType.FUNCTIONAL:
            funct_reqs += "description: " + req.description + "\n"

    ufs_json = extract_user_functions_prompt(funct_reqs)
    components_json = extract_user_function_components_prompt(funct_reqs, ufs_json)

    components = json.loads(components_json)

    ufp = 0

    for uft in ["ILF", "EIF", "EI", "EO", "EQ"]:
        referenced = "RET" if uft in ["ILF", "EIF"] else "FTR"
        for r in components[uft]:
            count_referenced = len(r[referenced])
            count_det = len(r["DET"])

            if count_det == 0:
                raise ValueError(f"{uft}: '{r['description']}' contains no DET.")

            if referenced == "RET" and count_referenced == 0:
                raise ValueError(f"{uft}: '{r['description']}' contains no RET.")

            x = get_index(count_referenced, complexity_table[uft][referenced])
            y = get_index(count_det, complexity_table[uft]["DET"])

            complexity = complexity_levels[x][y]

            ufp += complexity_weights[uft][complexity]

    return ufp


# COMPLEXITY TABLE

ILF_EIF = {
    "RET": [(1, 1), (2, 5), (6, float("inf"))],
    "DET": [(1, 19), (20, 50), (51, float("inf"))],
}

EO_EQ = {
    "FTR": [(0, 1), (2, 3), (4, float("inf"))],
    "DET": [(1, 5), (6, 19), (20, float("inf"))],
}

complexity_table = {
    "ILF": ILF_EIF,
    "EIF": ILF_EIF,
    "EO": EO_EQ,
    "EQ": EO_EQ,
    "EI": {
        "FTR": [(0, 1), (2, 2), (3, float("inf"))],
        "DET": [(1, 4), (5, 15), (16, float("inf"))],
    },
}

# 0 = LOW, 1 = AVG, 2 = HIGH
complexity_levels = [[0, 0, 1], [0, 1, 2], [1, 2, 2]]

complexity_weights = {
    "ILF": [7, 10, 15],
    "EIF": [5, 7, 10],
    "EI": [3, 4, 6],
    "EO": [4, 5, 7],
    "EQ": [3, 4, 6],
}


def get_index(value: int, tuples: list[tuple[int, int]]) -> int:
    index = 0
    for low, upper in tuples:
        if low <= value and value <= upper:
            return index
        else:
            index += 1
    raise ValueError(f"No index found for value: {value}")
