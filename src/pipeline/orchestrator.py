from pipeline.extraction import extract_requirements

def run_pipeline(text: str) -> str:
    result = extract_requirements(text)

    s = ""
    for r in result:
        s += f"id: {r.id}\n"
        s += f"title: {r.title}\n"
        s += f"description: {r.description}\n"
        s += f"type: {r.type.value}\n\n"

    return s