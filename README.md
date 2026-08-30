# AI-Based Effort Estimation from Software Requirements

## 1. Overview

This project estimates software development effort based on initial software requirements. It extracts and analyzes software requirements using a local small language model (SLM), estimates software size using Function Point Analysis, converts the resulting Unadjusted Function Points (UFP) to Source Lines of Code (SLOC), and applies COCOMO II together with an AI adjustment factor to estimate the effort in person-months.

### How it works

1. The initial software requirements are provided in `input.txt`.
2. The user selects the SLOC/UFP conversion factor for the target programming language.
3. The input is split into smaller chunks.
4. Software requirements are extracted from each chunk and combined into a single list.
5. Similar requirements are merged.
6. Unadjusted Function Points are estimated based on the extracted requirements. If insufficient information is available for certain Function Point components, fallback values are used and counted.
7. The COCOMO II breakage percentage is estimated.
8. The relevant COCOMO II scale and cost drivers are estimated.
9. The estimated Function Points are converted to SLOC using the selected conversion factor.
10. The development effort is calculated in person-months using COCOMO II.
11. An AI reduction factor is applied to the estimated effort.


## 2. Requirements

* Python 3.11 
    - tested with Python 3.11.9
* pip
* A GGUF language model that is compatible with llama.cpp
    - tested with Qwen3-14B-GGUF/Qwen3-14B-Q5_K_M.gguf
    - Available on Hugging Face: https://huggingface.co/Qwen/Qwen3-14B-GGUF?show_file_info=Qwen3-14B-Q5_K_M.gguf
* Hardware
    - Sufficient RAM and/or VRAM to load the selected model.

## 3. Project Structure

* `src/`

  * `data/`

    * `input.txt`

      - Input file containing the initial software requirements on which the program operates.

  * `estimation/`

    * `breakage_estimator.py`

      - Estimates the COCOMO II breakage percentage.

    * `scale_and_cost_driver_estimator.py`

      - Estimates the COCOMO II scale drivers PREC, FLEX, and RESL. Nominal values are used for TEAM and PMAT because the initial requirements do not provide sufficient information for the SLM to estimate them reliably.
      - Estimates the COCOMO II cost drivers RCPX, RUSE, and PDIF. Nominal multipliers are used for PERS, PREX, FCIL, and SCED because the initial requirements do not provide sufficient information for the SLM to estimate them reliably.
      - Estimates the AI reduction factor based on the COCOMO II cost driver CPLX.

    * `size_estimator.py`

      - Estimates the Unadjusted Function Points based on a list of software requirements.

  * `llm/`

    * `backend.py`

      - Handles the language model and sends all prompts to it.

  * `pipeline/`

    * `chunker.py`

      - Splits the input into chunks with a maximum size defined by `MAX_CHARS` in `config.py`.

    * `extractor.py`

      - Extracts software requirements from the input.

    * `merger.py`

      - Merges similar software requirements.

    * `orchestrator.py`

      - Runs and orchestrates the processing pipeline.

  * `schema/`

    * `languages.py`

      - Contains SLOC/UFP conversion factors for different programming languages and language categories.

    * `models.py`

      - Defines the classes used to represent software requirements.

  * `app.py`

    - Main entry point of the program.

  * `config.py`

    - Contains the program configuration.

* `tests/`

  * `chunker_test.py`

    - Contains pytest tests for `chunker.py`.

* `requirements.txt`

  - Contains the Python dependencies required by the program, which can be installed using pip.

## 4. Installation

1. Create and activate a Python virtual environment:

  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

2. Install the required Python packages:

  ```bash
  pip install -r requirements.txt
  ```

3. Download a llama.cpp-compatible GGUF model. The program was tested with:

  `Qwen3-14B-GGUF/Qwen3-14B-Q5_K_M.gguf`

  Available on Hugging Face:
  https://huggingface.co/Qwen/Qwen3-14B-GGUF?show_file_info=Qwen3-14B-Q5_K_M.gguf

4. Set the path to the downloaded model in `src/config.py`.

## 5. Configuration

The main configuration parameters can be found in `src/config.py`.

* `MODEL_PATH`
  - Path to the GGUF model file.
  - By default, the program expects the model at `models/Qwen3-14B-Q5_K_M.gguf`.
  - A custom path can be set using the `MODEL_PATH` environment variable:

    ```bash
    export MODEL_PATH="/path/to/model.gguf"
    ```

* `CONTEXT_SIZE`

  - Maximum context size used by the language model.

* `THREADS`

  - Number of CPU threads used for inference.

* `GPU_LAYERS = -1`

  - Number of model layers that are offloaded to the GPU.
  - `-1` attempts to offload all model layers to the GPU.
  - GPU offloading requires a llama-cpp-python build with GPU support.
  - Set this to `0` to run the model entirely on the CPU.

* `TEMPERATURE = 0.0`

  - Controls the randomness of the model output.
  - A value of `0.0` is used to make the output as deterministic as possible.

* `MAX_TOKENS = 2048`

  - Maximum number of tokens the model can generate for a single response.

### Chunking Configuration

* `TOKENS_PER_CHUNK = 250`

  - Approximate maximum number of tokens per input chunk.

* `CHARS_PER_TOKEN = 4`

  - Approximate number of characters per token used to convert the token limit into a character limit.

* `MAX_CHARS = TOKENS_PER_CHUNK * CHARS_PER_TOKEN`

  - Maximum size of an input chunk in characters.
  - With the default configuration, this results in a maximum chunk size of approximately `1000` characters.

## 6. Input Format

The input must be a UTF-8 encoded text file containing the project's initial software requirements.

The input file must be located at:

`src/data/input.txt`


## 7. Running the Application

Run the application from the project root directory:

```bash
python src/app.py
```

## 8. Example

When the application starts, the user is asked to select the primary programming language or a corresponding language category. The selected SLOC/UFP conversion factor is used to convert the estimated Unadjusted Function Points to Source Lines of Code.

Example:

```text
The program needs to convert Unadjusted Function Points to Source Lines of Code.
The factor is different for every programming language.

Please select the primary programming language used in your project by typing the number next to it.
If the language is not listed, you can choose a category or enter a custom factor.


Programming languages:

1) ABAP (SAP): 18

2) ASP: 54

3) Assembler: 98

4) Brio: 14

5) C: 99

6) C++: 53

7) C#: 59

8) COBOL: 55

9) Cognos Impromptu Scripts: 42

10) Cross System Products (CSP): 18

11) Cool:Gen/IEF: 24

12) Datastage: 65

13) Excel: 191

14) Focus: 45

15) FoxPro: 35

16) HTML: 40

17) J2EE: 49

18) Java: 53

19) JavaScript: 53

20) JCL: 48

21) LINC II: 30

22) Lotus Notes: 21

23) Natural: 34

24) .NET: 60

25) Oracle: 40

26) PACBASE: 32

27) Perl: 15

28) PL/I: 80

29) PL/SQL: 35

30) Powerbuilder: 28

31) REXX: 80

32) Sabretalk: 66

33) SAS: 37

34) Siebel: 60

35) SLOGAN: 75

36) SQL: 21

37) VB.NET: 60

38) Visual Basic: 44


Programming language categories:

39) Low-Level / System-Oriented Languages: 98.5

40) Classical Procedural / Legacy Languages: 62.0

41) Web / Enterprise Technologies: 51.2

42) Modern General-Purpose / Object-Oriented Languages: 53.8

43) 4GL / Business Tools / Application Generators: 44.7

44) Scripting / Automation Languages: 46.3

45) Database / Query / Reporting Languages: 35.6


Custom factor:

46) Custom factor


Select option: 41


Results:

Estimated effort in Person Months (PM): 2.2 PM

AI reduction factor: 10

Functions using fallback values: 0
```

## 9. Output

* `Estimated effort in Person Months (PM)`

  - The estimated software development effort in person-months after applying the AI reduction factor.
  
* `AI reduction factor`

  - The factor by which the COCOMO II effort estimate is divided to account for the expected productivity increase from AI-assisted development.
  - For example, a factor of 10 divides the original COCOMO II effort estimate by 10.

* `Functions using fallback values`

  - The number of user functions for which fallback values had to be used during Function Point estimation because the requirements did not provide sufficient information.
  - A high number of fallback values may indicate that the input requirements do not contain enough detail for a reliable Function Point estimation.

## 10. Running the Tests

Run the test suite from the project root directory using:

```bash
pytest
```

## 11. Known Limitations

1. **Requirement quality**
   The estimation depends heavily on the quality, completeness, and clarity of the provided requirements.

2. **LLM dependency**
   Results may vary depending on the language model, quantization, and model configuration used.

3. **LLM reliability**
   The model may produce incorrect, inconsistent, or hallucinated classifications and estimations.

4. **Limited context window**
   Large requirement documents must be split into chunks, which can cause relevant context to be lost.

5. **Chunking effects**
   Related requirements may be separated into different chunks, making dependencies or duplicates harder to detect.

6. **Requirement merging and normalization**
   Similar requirements may be merged incorrectly, while duplicates may remain undetected.

7. **Fixed COCOMO II parameters**
   Some COCOMO II scale and cost drivers are set to `Nominal` because they cannot be reliably derived from requirements alone.

8. **Missing team and organizational information**
   Developer experience, team capabilities, tool support, and organizational processes cannot be fully inferred from initial software requirements and are therefore only partially represented by the estimation.

9. **Heuristic AI reduction factor**
   The AI reduction factor is not an established part of COCOMO II and is not based on extensive empirical calibration. Since it directly affects the final effort estimate, different factor values can significantly change the result.
