# GPT-OSS-20B Speculative Decoding Benchmark

Inference-time benchmark for GPT-OSS-20B with EAGLE3, Arctic suffix decoding,
and DFlash. (benchmark-feb26 branch has comparison between Draft and EAGLE3)

## Dataset

BFCL v3 prompts from `gorilla-llm/Berkeley-Function-Calling-Leaderboard`:

- `BFCL_v3_simple.json`
- `BFCL_v3_multiple.json`
- `BFCL_v3_parallel.json`
- `BFCL_v3_parallel_multiple.json`

The first 200 rows are formatted as agent JSON tool-call generation requests.

## Code Layout

- `inference_eagle3.py` downloads the base and EAGLE3 draft models, starts vLLM
  with the EAGLE3 speculative config, runs both benchmark phases, and saves
  `results/eagle3.json`.
- `inference_suffix_decoding.py` downloads the base model, starts vLLM with
  Arctic suffix decoding, runs both benchmark phases, and saves
  `results/suffix_decoding.json`.
- `inference_dflash.py` downloads the base and DFlash draft models, starts vLLM
  with the DFlash speculative config, runs both benchmark phases, and saves
  `results/dflash.json`.
- `benchmark.py` loads BFCL prompts, sends OpenAI-compatible chat completion
  requests, and calculates latency/throughput metrics.

## Runs

Each method uses 8 warmup requests before measurement.

- Sequential: 200 requests, `--max-num-seqs 1`
- Parallel: 200 requests, concurrency 8, `--max-num-seqs 8`

## Results

| Method | Phase | Total time | Req/s | Completion tok/s | Mean latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|
| EAGLE3 | Sequential | 138.30s | 1.45 | 164.90 | 0.691s | 1.080s |
| EAGLE3 | Parallel | 26.26s | 7.62 | 837.93 | 1.022s | 1.676s |
| Suffix decoding | Sequential | 98.25s | 2.04 | 255.24 | 0.491s | 0.881s |
| Suffix decoding | Parallel | 17.51s | 11.42 | 1431.03 | 0.691s | 1.107s |
| DFlash | Sequential | 66.80s | 2.99 | 362.97 | 0.334s | 0.552s |
| DFlash | Parallel | 13.82s | 14.47 | 1776.10 | 0.538s | 1.017s |

![Benchmark summary](results/benchmark_summary.png)

## Backends

Backend selection from the saved vLLM startup logs:

| Method | Target attention | Target MoE | Draft attention |
|---|---|---|---|
| EAGLE3 | `TRITON_ATTN` | `MARLIN` Mxfp4 | `FLASH_ATTN` / FlashAttention v2 |
| Suffix decoding | `TRITON_ATTN` | `MARLIN` Mxfp4 | none |
| DFlash | `TRITON_ATTN` | `MARLIN` Mxfp4 | `FLASH_ATTN` / FlashAttention v2 |

The target GPT-OSS-20B model used `MoEPrepareAndFinalizeNoDPEPModular` for MoE
prepare/finalize. FlashInfer top-p/top-k sampling was disabled for all runs, so
sampling used the PyTorch-native sampler.

## Commands

Run one method:

```bash
cd spec-decoding-bench
python3 inference_eagle3.py
python3 inference_suffix_decoding.py
python3 inference_dflash.py
```

Run all methods:

```bash
cd spec-decoding-bench
python3 run_all.py
```

Create the summary graph:

```bash
cd spec-decoding-bench
python3 plot_results.py
```

## Notes

`VLLM_USE_FLASHINFER_SAMPLER=0` is set inside the vLLM subprocess because this
image has `curand.h` under the Python NVIDIA package path instead of
`/usr/local/cuda/include`, which breaks FlashInfer sampler JIT.

DFlash uses `--disable-hybrid-kv-cache-manager` so its draft attention layers
stay in a compatible KV cache group on GPT-OSS's hybrid sliding/full attention
layout.
