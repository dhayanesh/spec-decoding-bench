from pathlib import Path


BASE_MODEL = "openai/gpt-oss-20b"
SERVED_MODEL = "gpt-oss-20b"

EAGLE3_MODEL = "zhuyksir/EAGLE3-gpt-oss-20b-bf16"
DFLASH_MODEL = "jianchen0311/gpt-oss-20b-DFlash"

HOST = "127.0.0.1"
MAX_MODEL_LEN = 4096
MAX_TOKENS = 192
REQUESTS = 200
PARALLELISM = 8
WARMUP_REQUESTS = 8

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

COMMON_SERVER_ARGS = [
    "--served-model-name",
    SERVED_MODEL,
    "--host",
    HOST,
    "--max-model-len",
    str(MAX_MODEL_LEN),
    "--max-num-batched-tokens",
    "16384",
    "--gpu-memory-utilization",
    "0.90",
    "--no-enable-log-requests",
]
