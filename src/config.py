import os

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "models/Qwen3-14B-Q5_K_M.gguf",
)

CONTEXT_SIZE = 8192

# Use the number of CPUs assigned by SLURM, or 4 threads otherwise.
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

GPU_LAYERS = -1
TEMPERATURE = 0.0
MAX_TOKENS = 2048

TOKENS_PER_CHUNK = 250
CHARS_PER_TOKEN = 4
MAX_CHARS = TOKENS_PER_CHUNK * CHARS_PER_TOKEN
