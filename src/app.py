from pipeline.orchestrator import run_pipeline


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


text = read_file("src/data/input.txt")
result = run_pipeline(text)
print(f"Estimated size: {result} KSLOC\n")
