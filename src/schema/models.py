from enum import Enum
from pydantic import BaseModel


class RequirementType(str, Enum):
    """
    Represents the possible types of requirements.
    """

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"


class Requirement(BaseModel):
    """
    Represents a software requirement.

    Attributes:
        description: Description of the requirement.
        type: Type of the requirement.
    """

    description: str
    type: RequirementType
