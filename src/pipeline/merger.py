from llm.backend import merge_prompt
from schema.models import Requirement
import json


def merge_requirements(requirements: list[Requirement]) -> list[Requirement]:
    text = json.dumps([req.model_dump() for req in requirements], indent=2)
    # The following print statements in lines 9, 10, 12 and 13 are for debugging purposes and can be removed in production.
    print("------ MERGE SEND TO MODEL ------")
    print(text)
    response = merge_prompt(text)
    print("------ MODEL RESPONSE ------")
    print(response)
    json_obj = json.loads(response)
    return [Requirement(**req) for req in json_obj]
