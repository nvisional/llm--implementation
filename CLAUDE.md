# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repo contains two LLM projects:
- **nanochat/** — The main project: a full-stack, hackable ChatGPT-like system for training and inference on modest budgets (~$100–$1000). Designed for single 8×H100 node training.
- **minGPT/** — A simpler educational character-level GPT (Chinese-language comments, Jupyter notebook).

## Development Setup (nanochat)

Uses `uv` as the package manager with a `.venv/` directory.

```bash
cd nanochat
uv sync                          # install dependencies
maturin develop --release        # compile Rust BPE tokenizer (required before use)
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

Config overrides via CLI: `python -m scripts.base_train --depth=20 --device_batch_size=16 --run=my_run`

## Architecture

### Core Package (`nanochat/nanochat/`)

| File | Role |
|------|------|
| `gpt.py` | Transformer model: `GPTConfig`, `CausalSelfAttention`, `GPT` |
| `engine.py` | Inference engine with KV cache and token generation |
| `tokenizer.py` | BPE tokenizer (HuggingFace wrapper + `rustbpe` Rust bindings) |
| `dataloader.py` | Distributed tokenizing data loader |
| `adamw.py` / `muon.py` | Distributed AdamW and Muon optimizers |
| `configurator.py` | Python exec-based config system (alternative to argparse) |
| `common.py` | Logging, DDP setup, device detection utilities |
| `core_eval.py` | CORE benchmark evaluation (from DCLM paper) |
| `execution.py` | Python code execution tool for LLM tool use |

### Model Architecture Choices

- Rotary positional embeddings (no learned positional embeddings)
- QK normalization in attention
- ReLU² MLP activation
- RMSNorm with no learnable parameters
- No bias in linear layers
- Untied token embedding and lm_head weights
- Group-Query Attention (GQA) support

### Evaluation Tasks (`nanochat/tasks/`)

`arc.py`, `mmlu.py`, `gsm8k.py`, `humaneval.py`, `spellingbee.py`, `customjson.py`, `smoltalk.py` — all follow `TaskMixture`/`TaskSequence` base classes from `tasks/common.py`.

### Rust Component (`nanochat/rustbpe/`)

Custom BPE tokenizer in Rust compiled via Maturin/PyO3. Must be compiled before running tokenizer-dependent code.

## Key Design Patterns

- **Config system**: `configurator.py` uses `exec()` to load Python config files and `--key=value` CLI overrides with type validation. Scripts define defaults as module-level variables.
- **Distributed training**: All training scripts use `torchrun` and PyTorch DDP. The `common.py` module handles DDP initialization and rank-aware logging.
- **Checkpointing**: `checkpoint_manager.py` handles save/load; training scripts resume from the latest checkpoint automatically.
