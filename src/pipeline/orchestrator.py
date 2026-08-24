from pipeline.extractor import extract_requirements
from pipeline.chunker import chunk_text
from pipeline.merger import merge_requirements
from estimation.size_estimator import estimate_size
from estimation.breakage_estimator import estimate_breakage
from estimation.scale_and_cost_driver_estimator import (
    estimate_scaling_drivers,
    estimate_cost_drivers_and_ai_factor,
)
from schema.models import Requirement
import json


def run_pipeline(text: str, sloc_factor: float) -> tuple[float, int]:
    chunks = chunk_text(text)
    extracted = []

    for chunk in chunks:
        extracted.extend(extract_requirements(chunk))

    reqs = merge_requirements(extracted)
    normalized_reqs = normalize(reqs)

    size = estimate_size(normalized_reqs)

    text_reqs = json.dumps([req.model_dump() for req in reqs], indent=2)

    brak_percentage = estimate_breakage(text_reqs)
    if brak_percentage < 0 or brak_percentage > 100:
        raise ValueError(
            f"Breakage value must be between 0 and 100, received {brak_percentage}."
        )

    estimated_sloc = size[0] * sloc_factor * (1 + brak_percentage / 100)

    scale_exponent = estimate_scaling_drivers(text_reqs)
    cost_drivers, ai_reduction_factor = estimate_cost_drivers_and_ai_factor(text_reqs)

    cocomo_result = 2.94 * ((estimated_sloc / 1000) ** scale_exponent) * cost_drivers

    result = cocomo_result / ai_reduction_factor

    return result, size[1]


def normalize(requirements: list[Requirement]) -> list[Requirement]:
    result = []
    for req in requirements:
        req.description = req.description.strip()
        if req.description != "":
            result.append(req)

    return result
