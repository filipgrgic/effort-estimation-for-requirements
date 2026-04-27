from pipeline.extractor import extract_requirements
from pipeline.chunker import chunk_text
from pipeline.merger import merge_requirements
from estimation.size_estimator import estimate_size
from schema.models import Requirement


def run_pipeline(text: str) -> float:
    chunks = chunk_text(text)
    extracted = []

    for chunk in chunks:
        extracted.extend(extract_requirements(chunk))

    reqs = merge_requirements(extracted)
    normalized_reqs = normalize(reqs)

    size = estimate_size(normalized_reqs)

    return size


def normalize(requirements: list[Requirement]) -> list[Requirement]:
    result = []
    for req in requirements:
        req.description = req.description.strip()
        if req.description != "":
            result.append(req)

    return result
