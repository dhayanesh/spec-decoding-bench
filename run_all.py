import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "inference_eagle3.py",
    "inference_suffix_decoding.py",
    "inference_dflash.py",
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        subprocess.run([sys.executable, str(root / script)], check=True)
