# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a learning repository for becoming an AI engineer, covering the full stack: pretraining → inference → hardware acceleration. The goal is to read all source files, write annotated notes, and eventually run the full training pipeline on GPU.

**Structure:**
- **nanochat/** — git submodule pointing to [karpathy/nanochat](https://github.com/karpathy/nanochat). Full-stack ChatGPT-like training+inference system. Do not commit changes here; it tracks upstream.
- **minGPT/** — Simpler educational character-level GPT (Chinese-language comments, Jupyter notebook).
- **notes/** — Chinese annotated deep-dives into each source file (one file per module).
- **dev/** — Local dev scripts: debug launcher, single-GPU training script.

## Development Setup (nanochat)

Uses `uv` as the package manager with a `.venv/` directory.

```bash
cd nanochat
uv sync                                                        # install dependencies
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml  # compile Rust BPE tokenizer (required before use)
```

**Local MPS training (Mac)** — runs but slow, for pipeline verification only:
```bash
cd nanochat
uv run python3 -m scripts.base_train \
    --depth=4 --max-seq-len=64 --device-batch-size=1 \
    --total-batch-size=64 --num-iterations=3 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --window-pattern=L
```

**Debug with VS Code + debugpy** — attach on port 9501:
```bash
cd nanochat && uv run python3 ../dev/debug_train.py
# then F5 in VS Code with "Attach debugpy (port 9501)" config
```

## Running Tests

```bash
cd nanochat
python -m pytest tests/ -v                      # all tests
python -m pytest tests/ -m "not slow"           # skip slow tests
python -m pytest tests/test_engine.py -v -s     # single test file
```

## Training Pipeline

The full pipeline is automated in `speedrun.sh` (~4 hrs, 8×H100) or `run1000.sh` (~33 hrs, larger model). For CPU/MPS development, use `dev/runcpu.sh`.

Manual pipeline order:
1. `python -m scripts.tok_train` — train BPE tokenizer
2. `python -m scripts.base_train` — pretrain base model (torchrun for distributed)
3. `python -m scripts.base_eval` — evaluate on CORE metric
4. `python -m scripts.mid_train` — midtraining with conversation tokens
5. `python -m scripts.chat_sft` — supervised fine-tuning
6. `python -m scripts.chat_eval` — evaluate chat model
7. `python -m scripts.chat_web` — launch FastAPI web UI

Config overrides via CLI use hyphens: `uv run python3 -m scripts.base_train --depth=20 --device-batch-size=16 --run=my_run`

**Single GPU experience run** (A100, ~$3–5): `bash dev/run_single_gpu.sh` from inside `nanochat/`.

## Architecture

### Core Package (`nanochat/nanochat/`)

| File | Role | Notes covered |
|------|------|---------------|
| `gpt.py` | Transformer model: `GPTConfig`, `CausalSelfAttention`, `GPT` | ✅ `notes/01-gpt-architecture.md` |
| `engine.py` | Inference engine with KV cache and token generation | 🔲 next |
| `tokenizer.py` | BPE tokenizer (HuggingFace wrapper + `rustbpe` Rust bindings) | 🔲 |
| `dataloader.py` | Distributed tokenizing data loader | 🔲 |
| `optim.py` | Muon + AdamW optimizers (previously adamw.py/muon.py) | 🔲 |
| `common.py` | Logging, DDP setup, device detection utilities | 🔲 |
| `core_eval.py` | CORE benchmark evaluation (from DCLM paper) | 🔲 |
| `execution.py` | Python code execution tool for LLM tool use | 🔲 |

### Model Architecture Choices

- Rotary positional embeddings (RoPE) — no learned positional embeddings
- QK normalization in attention — prevents attention score explosion
- ReLU² MLP activation — sparser than GELU, faster
- RMSNorm with no learnable parameters
- No bias in linear layers; untied token embedding and lm_head weights
- Group-Query Attention (GQA) — fewer KV heads than Q heads, smaller KV cache
- Value Residual (ResFormer-style) — alternating layers mix in static token embeddings
- Logit softcap at ±15 via tanh — stabilizes early training
- Sliding window attention pattern (`SSSL`) — most layers attend locally, last layer full context
- `smear`: mixes previous token embedding into current position (cheap bigram info)
- `backout`: subtracts mid-layer residual before logit projection to remove low-level features

### Evaluation Tasks (`nanochat/tasks/`)

`arc.py`, `mmlu.py`, `gsm8k.py`, `humaneval.py`, `spellingbee.py`, `customjson.py`, `smoltalk.py` — all follow `TaskMixture`/`TaskSequence` base classes from `tasks/common.py`.

### Rust Component (`nanochat/rustbpe/`)

Custom BPE tokenizer in Rust compiled via Maturin/PyO3. Must be compiled before running tokenizer-dependent code.

## Key Design Patterns

- **CLI args use hyphens**: `--device-batch-size`, not `--device_batch_size`. Underscore args will fail.
- **Distributed training**: `torchrun --nproc_per_node=8 -m scripts.base_train` for multi-GPU. Single GPU: just `python -m scripts.base_train`.
- **Checkpointing**: `checkpoint_manager.py` saves to `~/.cache/nanochat/base_checkpoints/d{depth}/`. Resume with `--resume-from-step=N`.
- **Optimizer split**: Matrix weights (Attention/MLP) use Muon; embeddings + lm_head + scalars use AdamW. Configured in `GPT.setup_optimizer()` (`gpt.py:369`).
- **Meta device init**: `GPT.__init__` runs on meta device (shapes only, no data). Actual weight allocation happens in `to_empty()` + `init_weights()` (`base_train.py:146–151`).
- **nanochat submodule**: Never commit changes inside `nanochat/`. Use `git config submodule.nanochat.ignore untracked` to suppress VS Code noise from build artifacts.

## GPU Training Notes

- **H100 (Hopper)**: Flash Attention 3 + FP8 training supported. Speedrun: ~99 min on 8×H100.
- **A100**: Falls back to FA2, no FP8. ~4–5× slower than H100 for this workload; costs more overall.
- **Mac MPS**: No FA3, no FP8, no CUDA. Use `--window-pattern=L` to avoid SDPA sliding window warning. Good for pipeline verification and debugging only.
