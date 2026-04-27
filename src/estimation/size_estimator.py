from llm.backend import extract_user_functions_prompt
from llm.backend import determine_complexity_prompt
from schema.models import Requirement
import json


def estimate_size(reqs: list[Requirement]) -> float:
    funct_reqs = ""
    for req in reqs:
        # Only functional requirements are relevant for function points
        if req.description == "functional":
            funct_reqs += "description: " + req.description + "\n"

    ufs_json = extract_user_functions_prompt(funct_reqs)
    complexity_json = determine_complexity_prompt(funct_reqs, ufs)

    ufs = json.loads(ufs_json)
    complexity = json.loads(complexity_json)

    for uft in ["ILF", "EIF", "EI", "EO", "EQ"]:
        if uft == "ILF" or uft == "EIF":
            referenced = "RET"
        else:
            referenced = "FTR"
        count_referenced = 0
        count_det = 0
        for r in complexity[uft]:
            count_referenced += len(r[referenced])
            count_det += len(r["DET"])

        count_functions = len(ufs[uft])


# TODO: Implement table for complexity
