from enum import Enum
from pydantic import BaseModel

class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"

class Requirement(BaseModel):
    id: str
    title: str
    description: str
    type: RequirementType
