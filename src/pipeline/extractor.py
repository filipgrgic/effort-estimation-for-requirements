from llm.backend import extract_prompt
from schema.models import Requirement
import json


def extract_requirements(text: str) -> list[Requirement]:
    """
    Extracts software requirements from the given text.

    Args:
        text: Text to extract requirements from.

    Returns:
        A list of extracted software requirements.
    """
    # The following print statements in lines 8, 9, 11 and 12 are for debugging purposes and can be removed in production.
    # print("------ EXTRACT SEND TO MODEL ------")
    # print(text)
    response = extract_prompt(text)
    # print("------ MODEL RESPONSE ------")
    # print(response)
    json_obj = json.loads(response)
    return [Requirement(**req) for req in json_obj]
