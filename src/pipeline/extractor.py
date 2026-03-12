from llm.backend import extract_prompt
from schema.models import Requirement
import json

def extract_requirements(text: str) -> list[Requirement]:
    response = extract_prompt(text)
    json_obj = json.loads(response)
    return [Requirement(**req) for req in json_obj]