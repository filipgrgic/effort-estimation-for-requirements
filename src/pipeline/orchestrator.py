from pipeline.extractor import extract_requirements
from pipeline.chunker import chunk_text
from config import MAX_CHARS


def run_pipeline(text: str) -> str:
    chunks = chunk_text(text)
    result = []

    for chunk in chunks:
        result.extend(extract_requirements(chunk))

    s = ""
    for r in result:
        s += f"description: {r.description}\n"
        s += f"type: {r.type.value}\n\n"

    return s
