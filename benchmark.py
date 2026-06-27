import asyncio
import json
import statistics
import time
from typing import Any

import httpx
from huggingface_hub import hf_hub_download

from settings import DATA_DIR, MAX_TOKENS, PARALLELISM, REQUESTS, SERVED_MODEL


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_FILES = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
]


def build_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    functions = json.dumps(row["function"], separators=(",", ":"))
    question = row["question"][0][0]["content"]
    content = (
        "Available functions:\n"
        f"{functions}\n\n"
        "User request:\n"
        f"{question}\n\n"
        "Return only JSON with this shape: "
        '{"tool_calls":[{"name":"function.name","arguments":{}}]}.'
    )
    return [
        {
            "role": "system",
            "content": "You are an agent planner that emits valid JSON only.",
        },
        {"role": "user", "content": content},
    ]


def load_prompts(limit: int = REQUESTS) -> list[list[dict[str, str]]]:
    DATA_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    for name in BFCL_FILES:
        path = hf_hub_download(
            BFCL_REPO,
            name,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())

    prompts = [build_prompt(row) for row in rows[:limit]]
    if len(prompts) < limit:
        raise RuntimeError(f"BFCL only produced {len(prompts)} prompts")
    return prompts


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
        },
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()

    usage = (response.json().get("usage") or {})
    return {
        "latency_seconds": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def run_sequential(
    base_url: str,
    prompts: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        samples = []
        for prompt in prompts:
            samples.append(await send_request(client, base_url, prompt))
        return samples


async def run_parallel(
    base_url: str,
    prompts: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(PARALLELISM)

    async with httpx.AsyncClient(timeout=300.0) as client:

        async def run_one(prompt: list[dict[str, str]]) -> dict[str, Any]:
            async with semaphore:
                return await send_request(client, base_url, prompt)

        return await asyncio.gather(*(run_one(prompt) for prompt in prompts))


def summarize(
    method: str,
    phase: str,
    max_num_seqs: int,
    samples: list[dict[str, Any]],
    total_seconds: float,
) -> dict[str, Any]:
    latencies = [sample["latency_seconds"] for sample in samples]
    completion_tokens = sum(sample["completion_tokens"] for sample in samples)
    prompt_tokens = sum(sample["prompt_tokens"] for sample in samples)

    return {
        "method": method,
        "phase": phase,
        "requests": len(samples),
        "max_num_seqs": max_num_seqs,
        "total_seconds": total_seconds,
        "requests_per_second": len(samples) / total_seconds,
        "completion_tokens_per_second": completion_tokens / total_seconds
        if total_seconds
        else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": statistics.quantiles(latencies, n=20)[18],
            "min": min(latencies),
            "max": max(latencies),
        },
    }


async def run_phase(
    method: str,
    phase: str,
    base_url: str,
    max_num_seqs: int,
    prompts: list[list[dict[str, str]]],
) -> dict[str, Any]:
    started = time.perf_counter()
    if phase == "sequential":
        samples = await run_sequential(base_url, prompts)
    else:
        samples = await run_parallel(base_url, prompts)
    elapsed = time.perf_counter() - started
    return summarize(method, phase, max_num_seqs, samples, elapsed)
