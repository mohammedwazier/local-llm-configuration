# Baseline Benchmark

Captured from the actual running system on 2026-05-06.

## Hardware

| Component | Spec |
|---|---|
| CPU | Intel Core i5-11400F @ 2.60GHz (6C / 12T) |
| GPU | NVIDIA GeForce RTX 3060 12 GB GDDR6 |
| RAM | 64 GB DDR4 |
| Storage | 468 GB NVMe |
| OS | Linux Mint 22.3 (Ubuntu 24.04) |

## Software

| Component | Version |
|---|---|
| Backend | llama-cpp-turboquant (commit 69d8e4b) |
| GPU Driver | NVIDIA 580.126.09 (CUDA 12) |

## Model

| Property | Value |
|---|---|
| Model | Qwen3.6-35B-A3B (MoE) |
| Quant | UD-Q4_K_M (22.1 GB file, 8.4 GB active) |
| Architecture | 40 target layers, 8 shared KV layers |
| Active params | ~3B per token |
| Context | 262,144 tokens |
| VRAM model | 4,309 MiB (CUDA0) |
| VRAM KV cache | 1,180 MiB (turbo4 K + turbo3 V) |
| VRAM total | 7,685 MiB / 12,288 MiB |

## Speculative Decoding

| Setting | Value |
|---|---|
| Type | `ngram-mod` |
| Draft tokens | 8 |
| No separate draft model needed | |

## Performance

**Prompt (prefill):** 88.3 tok/s
**Generation (decode):** 27.2 tok/s
**Latency:** ~7.7s for 200 tokens

Benchmark command:
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Write a short paragraph about the Rust programming language, focusing on its memory safety features. Keep it under 100 words."}],"max_tokens":200,"temperature":0}'
```

## Service Configuration

| File | Purpose |
|---|---|
| `llama-qwen.service` | llama-server with turboquant + speculative decoding |
| `rag-server.service` | RAG proxy with ChromaDB + feedback loop |
| `docker-compose.yml` | ChromaDB container |

Key flags:
```
-ngl 999              # offload all layers to GPU
-ncmoe 35             # 35 MoE expert layers on CPU
-fa on                # Flash Attention
--cache-type-k turbo4 # 4-bit TurboQuant K cache
--cache-type-v turbo3 # 3-bit TurboQuant V cache
--no-mmap --mlock     # pin model in RAM
-c 262144             # 256K context window
--spec-type ngram-mod # model-free speculative decoding
--draft-n 8           # 8 draft tokens
```
