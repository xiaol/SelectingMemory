"""
Run a repeated Raven vs RWKV-7 CUDA comparison matrix.

Example:
    PYTHONPATH=/home/xiaol/X/HRM-Text:$PYTHONPATH \
    LT2_RWKV7_CUDA_DIR=/home/xiaol/X/LT2_upstream/apps/LT2/cuda/rwkv7 \
    /home/xiaol/X/HRM-Text/.venv/bin/python examples/run_compare_matrix.py \
      --seq-lens 512 1024 2048 4096 \
      --modes forward backward \
      --repeats 3
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

import compare_mixers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward-warmup", type=int, default=2)
    parser.add_argument("--backward-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["forward", "backward"], choices=["forward", "backward"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--rwkv-backend", default="cuda", choices=["cuda", "auto", "torch"])
    parser.add_argument("--rwkv-head-size", type=int, default=64)
    parser.add_argument("--rwkv-chunk-len", type=int, default=16)
    parser.add_argument("--routed-rwkv-route-floor", type=float, default=0.1)
    parser.add_argument("--low-rank-slot-rwkv-rank", type=int, default=8)
    parser.add_argument("--low-rank-slot-rwkv-backend", default="auto", choices=["auto", "triton", "torch"])
    parser.add_argument(
        "--mixers",
        nargs="+",
        default=["raven", "rwkv7"],
        choices=["raven", "rwkv7", "routed_rwkv7", "slot_rwkv7", "low_rank_slot_rwkv7"],
    )
    parser.add_argument("--lt2-wrapper-root", type=Path, default=None)
    parser.add_argument("--lt2-cuda-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/matrix"))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _make_compare_args(args: argparse.Namespace, seq_len: int, mode: str, repeat: int) -> argparse.Namespace:
    backward = mode == "backward"
    return argparse.Namespace(
        mixers=list(args.mixers),
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        seq_len=seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_slots=args.num_slots,
        topk=args.topk,
        vocab_size=args.vocab_size,
        warmup=args.backward_warmup if backward else args.warmup,
        steps=args.backward_steps if backward else args.steps,
        backward=backward,
        seed=args.seed + repeat,
        rwkv_backend=args.rwkv_backend,
        rwkv_head_size=args.rwkv_head_size,
        rwkv_chunk_len=args.rwkv_chunk_len,
        routed_rwkv_route_floor=args.routed_rwkv_route_floor,
        low_rank_slot_rwkv_rank=args.low_rank_slot_rwkv_rank,
        low_rank_slot_rwkv_backend=args.low_rank_slot_rwkv_backend,
        lt2_wrapper_root=args.lt2_wrapper_root,
        lt2_cuda_dir=args.lt2_cuda_dir,
        json_out=None,
    )


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(row[key] for row in rows))


def _summarize(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in results:
        groups.setdefault((row["mode"], row["seq_len"], row["mixer"]), []).append(row)

    summary = []
    for (mode, seq_len, mixer), rows in sorted(groups.items()):
        summary.append(
            {
                "mode": mode,
                "seq_len": seq_len,
                "mixer": mixer,
                "repeats": len(rows),
                "params_m": _median(rows, "params_m"),
                "tokens_per_second_median": _median(rows, "tokens_per_second"),
                "tokens_per_second_min": min(row["tokens_per_second"] for row in rows),
                "tokens_per_second_max": max(row["tokens_per_second"] for row in rows),
                "peak_memory_gb_median": _median(rows, "peak_memory_gb"),
                "seconds_median": _median(rows, "seconds"),
            }
        )
    return summary


def _write_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "# Raven vs RWKV-7 Matrix",
        "",
        "| mode | seq len | mixer | repeats | params M | median tok/s | min tok/s | max tok/s | median peak GB | median seconds |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {row['seq_len']} | {row['mixer']} | {row['repeats']} | "
            f"{row['params_m']:.2f} | {row['tokens_per_second_median']:.0f} | "
            f"{row['tokens_per_second_min']:.0f} | {row['tokens_per_second_max']:.0f} | "
            f"{row['peak_memory_gb_median']:.3f} | {row['seconds_median']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    compare_mixers._maybe_prepend_sys_path(compare_mixers._repo_root())
    compare_mixers._configure_lt2_paths(args)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required to run the matrix benchmark.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    all_results = []
    for mode in args.modes:
        for seq_len in args.seq_lens:
            for repeat in range(args.repeats):
                run_args = _make_compare_args(args, seq_len, mode, repeat)
                compare_mixers._validate_cuda_rwkv_args(run_args)
                print(f"\nmode={mode} seq_len={seq_len} repeat={repeat + 1}/{args.repeats}")
                for mixer in run_args.mixers:
                    print(f"Benchmarking {mixer}...")
                    row = compare_mixers._bench_one(deepcopy(run_args), mixer, torch)
                    row["mode"] = mode
                    row["repeat"] = repeat
                    all_results.append(row)

    summary = _summarize(all_results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "compare_matrix.json"
    summary_path = args.out_dir / "compare_matrix_summary.md"
    result_path.write_text(
        json.dumps(
            {
                "args": compare_mixers._jsonable_args(args),
                "results": all_results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(summary_path, summary)
    print(f"\nWrote {result_path}")
    print(f"Wrote {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
