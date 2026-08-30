from pathlib import Path
from pipeline.orchestrator import run_pipeline
from schema.languages import languages, categories


def read_file(filepath: str | Path) -> str:
    """
    Reads the contents of a file.

    Args:
        filepath: Path to the file to read.

    Returns:
        The contents of the file as a string.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def start_terminal() -> float:
    """
    Prompts the user to select an SLOC/UFP conversion factor.

    Returns:
        The selected SLOC/UFP factor.

    Raises:
        ValueError: If the input is invalid or the selected factor is not greater than 0.
    """
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


def main() -> None:
    """
    Runs the effort estimation pipeline and prints the results.
    """
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_FILE = BASE_DIR / "data" / "input.txt"
    text = read_file(INPUT_FILE)
    sf = start_terminal()
    result = run_pipeline(text, sf)
    print(f"\nEstimated effort in Person Months (PM): {result[0]:.3f} PM\n")
    print(f"AI reduction factor: {result[1]}\n")
    print(f"Functions using fallback values: {result[2]}\n")


if __name__ == "__main__":
    main()
