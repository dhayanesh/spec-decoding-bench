import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download

from benchmark import load_prompts, run_phase, run_sequential
from settings import (
    BASE_MODEL,
    COMMON_SERVER_ARGS,
    EAGLE3_MODEL,
    HOST,
    PARALLELISM,
    RESULTS_DIR,
    WARMUP_REQUESTS,
)


METHOD = "eagle3"
PORT = 8101
SPEC_CONFIG = {
    "method": "eagle3",
    "model": EAGLE3_MODEL,
    "num_speculative_tokens": int(os.getenv("EAGLE3_NUM_SPECULATIVE_TOKENS", "6")),
}


def download_models() -> None:
    for model in [BASE_MODEL, EAGLE3_MODEL]:
        print(f"Downloading {model}")
        snapshot_download(model)


def server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def server_command(max_num_seqs: int) -> list[str]:
    return [
        "vllm",
        "serve",
        BASE_MODEL,
        "--port",
        str(PORT),
        "--max-num-seqs",
        str(max_num_seqs),
        "--speculative-config",
        json.dumps(SPEC_CONFIG, separators=(",", ":")),
        *COMMON_SERVER_ARGS,
    ]


async def wait_for_server(
    base_url: str,
    log_path: Path,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 900
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = log_path.read_text(errors="ignore")[-4000:]
                raise RuntimeError(
                    f"Server exited with code {process.returncode}. Log tail:\n{tail}"
                )
            try:
                response = await client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise RuntimeError(f"Server did not become ready. Log: {log_path}")


async def benchmark_with_server(
    phase: str,
    max_num_seqs: int,
    prompts: list[list[dict[str, str]]],
) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"{METHOD}_{phase}.log"
    base_url = f"http://{HOST}:{PORT}"

    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command(max_num_seqs),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env(),
        )

    try:
        await wait_for_server(base_url, log_path, process)
        await run_sequential(base_url, prompts[:WARMUP_REQUESTS])
        result = await run_phase(METHOD, phase, base_url, max_num_seqs, prompts)
        result["server_log"] = str(log_path)
        return result
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


async def main() -> None:
    prompts = load_prompts()
    download_models()

    results = [
        await benchmark_with_server("sequential", 1, prompts),
        await benchmark_with_server("parallel", PARALLELISM, prompts),
    ]

    out_path = RESULTS_DIR / f"{METHOD}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"result_file={out_path}")


if __name__ == "__main__":
    asyncio.run(main())
