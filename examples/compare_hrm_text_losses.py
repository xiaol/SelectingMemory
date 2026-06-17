"""
Train Raven and low-rank slot RWKV on the local HRM-Text token dataset and compare held-out loss.

This uses a simple causal-LM objective over the prepared HRM-Text tokens.bin stream.
It is intended for local architecture comparison, not as the full HRM PrefixLM/flame recipe.

Example:
    PYTHONPATH=/home/xiaol/X/HRM-Text:/home/xiaol/X/raven:$PYTHONPATH \\
    LT2_RWKV7_CUDA_DIR=/home/xiaol/X/LT2_upstream/apps/LT2/cuda/rwkv7 \\
    /home/xiaol/X/HRM-Text/.venv/bin/python examples/compare_hrm_text_losses.py \\
      --dataset-path /home/xiaol/X/hrm_text_subset_1B \\
      --steps 500 --eval-every 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("/home/xiaol/X/hrm_text_subset_1B"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/hrm_text_loss_compare_raven_vs_low_rank_slot_rwkv7"))
    parser.add_argument("--mixers", nargs="+", default=["raven", "low_rank_slot_rwkv7"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--low-rank-slot-rwkv-rank", type=int, default=8)
    parser.add_argument("--low-rank-slot-rwkv-backend", default="triton_fused")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    return torch.bfloat16 if dtype_name == "bf16" else torch.float32


def _load_tokens(dataset_path: Path) -> tuple[np.ndarray, dict]:
    metadata = json.loads((dataset_path / "metadata.json").read_text())
    dtype = np.dtype(metadata.get("token_dtype", "uint16"))
    tokens_path = dataset_path / "tokens.bin"
    if not tokens_path.exists():
        raise FileNotFoundError(tokens_path)
    tokens = np.memmap(tokens_path, mode="r", dtype=dtype)
    return tokens, metadata


def _sample_starts(
    rng: np.random.Generator,
    *,
    low: int,
    high: int,
    num_batches: int,
    batch_size: int,
) -> np.ndarray:
    if high <= low:
        raise ValueError(f"Invalid sample range: low={low}, high={high}")
    return rng.integers(low=low, high=high, size=(num_batches, batch_size), dtype=np.int64)


def _make_batch(tokens: np.ndarray, starts: np.ndarray, seq_len: int, device: torch.device) -> torch.Tensor:
    batch = np.stack([np.asarray(tokens[start : start + seq_len], dtype=np.int64) for start in starts])
    return torch.from_numpy(batch).to(device=device, non_blocking=True)


def _build_model(args: argparse.Namespace, mixer: str, vocab_size: int, dtype: torch.dtype, device: torch.device):
    from raven.models.raven import RavenConfig, RavenForCausalLM

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = RavenConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        num_slots=args.num_slots,
        topk=min(args.topk, args.num_slots),
        max_position_embeddings=args.seq_len,
        vocab_size=vocab_size,
        fuse_cross_entropy=False,
        add_gumbel_noise=False,
        sequence_mixer=mixer,
        rwkv7_backend="cuda",
        rwkv7_head_size=64,
        rwkv7_chunk_len=16,
        low_rank_slot_rwkv7_rank=args.low_rank_slot_rwkv_rank,
        low_rank_slot_rwkv7_backend=args.low_rank_slot_rwkv_backend,
    )
    return RavenForCausalLM(config).to(device=device, dtype=dtype)


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _evaluate(
    model,
    tokens: np.ndarray,
    val_starts: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for starts in val_starts:
        input_ids = _make_batch(tokens, starts, seq_len, device)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        losses.append(float(loss.detach().float().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _train_one(
    args: argparse.Namespace,
    mixer: str,
    tokens: np.ndarray,
    train_starts: np.ndarray,
    val_starts: np.ndarray,
    vocab_size: int,
) -> dict:
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = _build_model(args, mixer, vocab_size, dtype, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    history = []
    start_time = time.perf_counter()
    initial_val = _evaluate(model, tokens, val_starts, args.seq_len, device)
    print(f"{mixer} step=000 val_loss={initial_val:.4f}", flush=True)

    for step, starts in enumerate(train_starts, start=1):
        scale = _lr_scale(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale

        input_ids = _make_batch(tokens, starts, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        record = {
            "step": step,
            "train_loss": float(loss.detach().float().cpu()),
            "lr": args.lr * scale,
            "grad_norm": float(grad_norm.detach().float().cpu()),
        }
        if step % args.eval_every == 0 or step == args.steps:
            record["val_loss"] = _evaluate(model, tokens, val_starts, args.seq_len, device)
            print(
                f"{mixer} step={step:04d} train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f}",
                flush=True,
            )
        history.append(record)

    elapsed = time.perf_counter() - start_time
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    final_val = next(row["val_loss"] for row in reversed(history) if "val_loss" in row)
    result = {
        "mixer": mixer,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "final_train_loss": history[-1]["train_loss"],
        "best_val_loss": min([initial_val] + [row["val_loss"] for row in history if "val_loss" in row]),
        "elapsed_seconds": elapsed,
        "tokens_per_second": args.steps * args.batch_size * args.seq_len / elapsed,
        "peak_memory_gb": peak_gb,
        "history": history,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _write_summary(path: Path, setup: dict, results: list[dict]) -> None:
    eval_steps = sorted({0} | {row["step"] for result in results for row in result["history"] if "val_loss" in row})
    by_mixer = {result["mixer"]: result for result in results}
    lines = [
        "# HRM-Text Loss Comparison",
        "",
        "Objective: causal LM over the local HRM-Text token stream.",
        "",
        (
            f"Shape: hidden={setup['hidden_size']}, layers={setup['num_layers']}, "
            f"batch={setup['batch_size']}, seq={setup['seq_len']}, vocab={setup['vocab_size']}"
        ),
        "",
        "| mixer | params M | initial val | final val | best val | final train | tok/s | peak GB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['mixer']} | {result['params_m']:.2f} | {result['initial_val_loss']:.4f} | "
            f"{result['final_val_loss']:.4f} | {result['best_val_loss']:.4f} | "
            f"{result['final_train_loss']:.4f} | {result['tokens_per_second']:.0f} | "
            f"{result['peak_memory_gb']:.3f} |"
        )

    lines += ["", "| step | " + " | ".join(by_mixer) + " |", "| ---: | " + " | ".join("---:" for _ in by_mixer) + " |"]
    for step in eval_steps:
        values = []
        for result in by_mixer.values():
            if step == 0:
                values.append(result["initial_val_loss"])
            else:
                values.append(next(row["val_loss"] for row in result["history"] if row.get("step") == step))
        lines.append("| " + str(step) + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens, metadata = _load_tokens(args.dataset_path)
    vocab_size = int(metadata["tokenizer_info"]["vocab_size"])
    train_limit = int(len(tokens) * args.train_fraction)
    max_start = len(tokens) - args.seq_len - 1
    rng = np.random.default_rng(args.seed)
    train_starts = _sample_starts(
        rng,
        low=0,
        high=min(train_limit, max_start),
        num_batches=args.steps,
        batch_size=args.batch_size,
    )
    val_starts = _sample_starts(
        rng,
        low=train_limit,
        high=max_start,
        num_batches=args.eval_batches,
        batch_size=args.batch_size,
    )

    setup = {
        **vars(args),
        "dataset_path": str(args.dataset_path),
        "out_dir": str(args.out_dir),
        "vocab_size": vocab_size,
        "num_tokens": int(len(tokens)),
        "train_limit": train_limit,
        "objective": "causal_lm_token_stream",
    }
    results = [_train_one(args, mixer, tokens, train_starts, val_starts, vocab_size) for mixer in args.mixers]
    payload = {"setup": setup, "results": results}
    (args.out_dir / "loss_compare.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_summary(args.out_dir / "loss_summary.md", setup, results)
    print((args.out_dir / "loss_summary.md").read_text())


if __name__ == "__main__":
    main()
