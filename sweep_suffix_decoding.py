import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import snapshot_download

from benchmark import load_prompts, run_phase, run_sequential
from settings import (
    BASE_MODEL,
    COMMON_SERVER_ARGS,
    HOST,
    PARALLELISM,
    RESULTS_DIR,
    WARMUP_REQUESTS,
)


SWEEP_REQUESTS = 48
PORT = 8111
SWEEP_CONFIGS = [
    {
        "label": "suffix_t8_p005_f1",
        "method": "suffix",
        "num_speculative_tokens": 8,
        "suffix_decoding_max_tree_depth": 8,
        "suffix_decoding_max_cached_requests": 100000,
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.05,
    },
    {
        "label": "suffix_t12_p005_f1",
        "method": "suffix",
        "num_speculative_tokens": 12,
        "suffix_decoding_max_tree_depth": 12,
        "suffix_decoding_max_cached_requests": 100000,
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.05,
    },
    {
        "label": "suffix_t16_p01_f1",
        "method": "suffix",
        "num_speculative_tokens": 16,
        "suffix_decoding_max_tree_depth": 16,
        "suffix_decoding_max_cached_requests": 100000,
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.1,
    },
    {
        "label": "suffix_t16_p005_f15",
        "method": "suffix",
        "num_speculative_tokens": 16,
        "suffix_decoding_max_tree_depth": 16,
        "suffix_decoding_max_cached_requests": 100000,
        "suffix_decoding_max_spec_factor": 1.5,
        "suffix_decoding_min_token_prob": 0.05,
    },
    {
        "label": "suffix_t24_p01_f1",
        "method": "suffix",
        "num_speculative_tokens": 24,
        "suffix_decoding_max_tree_depth": 24,
        "suffix_decoding_max_cached_requests": 100000,
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.1,
    },
]


def download_models() -> None:
    print(f"Downloading {BASE_MODEL}")
    snapshot_download(BASE_MODEL)


def server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def server_command(spec_config: dict[str, Any]) -> list[str]:
    return [
        "vllm",
        "serve",
        BASE_MODEL,
        "--port",
        str(PORT),
        "--max-num-seqs",
        str(PARALLELISM),
        "--speculative-config",
        json.dumps(
            {k: v for k, v in spec_config.items() if k != "label"},
            separators=(",", ":"),
        ),
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


async def run_one_config(
    spec_config: dict[str, Any],
    prompts: list[list[dict[str, str]]],
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(exist_ok=True)
    label = spec_config["label"]
    log_path = RESULTS_DIR / f"sweep_{label}.log"
    base_url = f"http://{HOST}:{PORT}"

    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command(spec_config),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env(),
        )

    try:
        await wait_for_server(base_url, log_path, process)
        await run_sequential(base_url, prompts[:WARMUP_REQUESTS])
        result = await run_phase(
            "suffix_decoding",
            label,
            base_url,
            PARALLELISM,
            prompts[:SWEEP_REQUESTS],
        )
        result["spec_config"] = {
            k: v for k, v in spec_config.items() if k != "label"
        }
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

    results = []
    for spec_config in SWEEP_CONFIGS:
        print(f"Running sweep config {spec_config['label']}")
        results.append(await run_one_config(spec_config, prompts))

    out_path = RESULTS_DIR / "suffix_config_sweep.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"result_file={out_path}")


if __name__ == "__main__":
    asyncio.run(main())
