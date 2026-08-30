from pipeline.extractor import extract_requirements
from pipeline.chunker import chunk_text
from pipeline.merger import merge_requirements
from estimation.size_estimator import estimate_size
from estimation.breakage_estimator import estimate_breakage
from estimation.scale_and_cost_driver_estimator import (
    estimate_scale_drivers,
    estimate_cost_drivers_and_ai_factor,
)
from schema.models import Requirement
import json


def run_pipeline(text: str, sloc_factor: float) -> tuple[float, int, int]:
    """
    Runs the effort estimation pipeline.

    Args:
        text: Input text containing the software requirements.
        sloc_factor: SLOC/UFP conversion factor.

    Returns:
        A tuple containing the effort estimation in person-months, the AI reduction factor,
        and the number of fallback values used.

    Raises:
        ValueError: If the estimated breakage percentage is outside the range from 0 to 100.
    """
    chunks = chunk_text(text)
    extracted = []

    for chunk in chunks:
        extracted.extend(extract_requirements(chunk))

    reqs = merge_requirements(extracted)
    normalized_reqs = normalize(reqs)

    ufp, fallback_count = estimate_size(normalized_reqs)

    text_reqs = json.dumps([req.model_dump() for req in normalized_reqs], indent=2)

    brak_percentage = estimate_breakage(text_reqs)
    if brak_percentage < 0 or brak_percentage > 100:
        raise ValueError(
            f"Breakage value must be between 0 and 100, received {brak_percentage}."
        )

    estimated_sloc = ufp * sloc_factor * (1 + brak_percentage / 100)

    scale_exponent = estimate_scale_drivers(text_reqs)
    cost_drivers, ai_reduction_factor = estimate_cost_drivers_and_ai_factor(text_reqs)

    cocomo_result = 2.94 * ((estimated_sloc / 1000) ** scale_exponent) * cost_drivers

    result = cocomo_result / ai_reduction_factor

    return result, ai_reduction_factor, fallback_count


def normalize(requirements: list[Requirement]) -> list[Requirement]:
    """
    Normalizes a list of requirements by trimming descriptions and removing empty entries.

    Args:
        requirements: List of requirements to normalize.

    Returns:
        A normalized list of requirements.
    """
    result = []
    for req in requirements:
        req.description = req.description.strip()
        if req.description != "":
            result.append(req)

    return result
