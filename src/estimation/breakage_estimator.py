from schema.models import Requirement
from llm.backend import estimate_breakage_prompt
import json


def estimate_breakage(reqs: str) -> float:
    response = json.loads(estimate_breakage_prompt(reqs))
    return float(response["breakage"])
