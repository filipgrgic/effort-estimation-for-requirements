from schema.models import Requirement
from llm.backend import estimate_breakage_prompt
import json


def estimate_breakage(reqs: list[Requirement]) -> float:
    text = json.dumps([req.model_dump() for req in reqs], indent=2)
    response = json.loads(estimate_breakage_prompt(text))
    return float(response["breakage"])
