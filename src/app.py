from pathlib import Path
from pipeline.orchestrator import run_pipeline
from schema.languages import languages, categories


def read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def start_terminal() -> float:
    print(f"""
The program needs to convert Unadjusted Function Points to Source Lines of Code.
The factor is different for every programming language.

Please select the primary programming language used in your project by typing the number next to it.
If the language is not listed, you can choose a category or enter a custom factor.
""")
    print("\nProgramming languages:\n")
    options = []
    index = 1
    for language, factor in languages.items():
        options.append(factor)
        print(f"{index}) {language}: {factor}\n")
        index += 1

    print("\nProgramming language categories:\n")
    for category, factor in categories.items():
        options.append(factor)
        print(f"{index}) {category}: {factor}\n")
        index += 1

    print("\nCustom factor:\n")
    print(f"{index}) Custom factor\n")

    try:
        choice = int(input("\nSelect option: "))
    except ValueError:
        raise ValueError("Invalid input. Please enter a number.")

    if 1 <= choice <= len(options):
        return options[choice - 1]
    elif choice == index:
        try:
            custom = float(input("\nType in a custom SLOC/UFP factor: "))
        except ValueError:
            raise ValueError("Invalid input. Please enter a number.")

        if custom <= 0:
            raise ValueError("The custom factor must be greater than 0.")

        return custom

    raise ValueError("Invalid selection.")


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "input.txt"
text = read_file(INPUT_FILE)
sf = start_terminal()
result = run_pipeline(text, sf) / 1000
print(f"Estimated size: {result} KSLOC\n")

