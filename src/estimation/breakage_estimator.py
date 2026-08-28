from schema.models import Requirement
from llm.backend import estimate_breakage_prompt
import json


def estimate_breakage(reqs: str) -> float:
    """
    Estimate the COCOMO II breakage percentage.

    Args:
        reqs: JSON string containing the software requirements to analyze.

    Returns:
        The estimated breakage percentage.
    """
    response = json.loads(estimate_breakage_prompt(reqs))
    return float(response["breakage"])
