from pipeline.extractor import extract_requirements
from pipeline.chunker import chunk_text
from pipeline.merger import merge_requirements
from schema.models import Requirement


def run_pipeline(text: str) -> list[Requirement]:
    chunks = chunk_text(text)
    extracted = []

    for chunk in chunks:
        extracted.extend(extract_requirements(chunk))

    result = merge_requirements(extracted)

    return normalize(result)


def normalize(requirements: list[Requirement]) -> list[Requirement]:
    result = []
    for req in requirements:
        req.description = req.description.strip()
        if req.description != "":
            result.append(req)

    return result
