"""
Compare Raven's default routing-memory mixer against the RWKV-7 mixer.

Example:
    PYTHONPATH=../HRM-Text:$PYTHONPATH \
    LT2_RWKV7_CUDA_DIR=../LT2_upstream/apps/LT2/cuda/rwkv7 \
    python examples/compare_mixers.py --rwkv-backend cuda --dtype bf16 --backward
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_prepend_sys_path(path: Path | None) -> None:
    if path is None:
        return
    path = path.resolve()
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _configure_lt2_paths(args: argparse.Namespace) -> None:
    root = _repo_root()

    lt2_wrapper_root = args.lt2_wrapper_root
    if lt2_wrapper_root is None:
        env_root = os.environ.get("LT2_WRAPPER_ROOT")
        lt2_wrapper_root = Path(env_root) if env_root else root.parent / "HRM-Text"
    _maybe_prepend_sys_path(lt2_wrapper_root)

    if args.lt2_cuda_dir is not None:
        os.environ["LT2_RWKV7_CUDA_DIR"] = str(args.lt2_cuda_dir.resolve())
    elif "LT2_RWKV7_CUDA_DIR" not in os.environ:
        candidate = root.parent / "LT2_upstream" / "apps" / "LT2" / "cuda" / "rwkv7"
        if candidate.exists():
            os.environ["LT2_RWKV7_CUDA_DIR"] = str(candidate.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixers",
        nargs="+",
        default=["raven", "rwkv7"],
        choices=["raven", "rwkv7", "routed_rwkv7"],
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--backward", action="store_true", help="Benchmark loss backward pass, not just forward.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rwkv-backend", default="cuda", choices=["cuda", "auto", "torch"])
    parser.add_argument("--rwkv-head-size", type=int, default=64)
    parser.add_argument("--rwkv-chunk-len", type=int, default=16)
    parser.add_argument("--routed-rwkv-route-floor", type=float, default=0.1)
    parser.add_argument("--lt2-wrapper-root", type=Path, default=None, help="Repo root containing apps/LT2/rwkv7_cuda.py.")
    parser.add_argument("--lt2-cuda-dir", type=Path, default=None, help="Directory containing LT2 RWKV-7 CUDA sources.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _torch_dtype(torch, dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def _validate_cuda_rwkv_args(args: argparse.Namespace) -> None:
    if not any(mixer in {"rwkv7", "routed_rwkv7"} for mixer in args.mixers) or args.rwkv_backend != "cuda":
        return
    if args.device != "cuda":
        raise ValueError("RWKV backend 'cuda' requires --device cuda.")
    if args.dtype != "bf16":
        raise ValueError("LT2 RWKV-7 CUDA kernels require --dtype bf16.")
    if args.rwkv_head_size != 64:
        raise ValueError("LT2 RWKV-7 CUDA wrapper currently requires --rwkv-head-size 64.")
    if args.rwkv_chunk_len != 16:
        raise ValueError("LT2 RWKV-7 CUDA wrapper currently requires --rwkv-chunk-len 16.")


def _build_config(args: argparse.Namespace, mixer: str):
    from raven.models.raven import RavenConfig

    return RavenConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        num_slots=args.num_slots,
        topk=min(args.topk, args.num_slots),
        max_position_embeddings=args.seq_len,
        vocab_size=args.vocab_size,
        fuse_cross_entropy=False,
        sequence_mixer=mixer,
        rwkv7_backend=args.rwkv_backend,
        rwkv7_head_size=args.rwkv_head_size,
        rwkv7_chunk_len=args.rwkv_chunk_len,
        routed_rwkv7_route_floor=args.routed_rwkv_route_floor,
    )


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench_one(args: argparse.Namespace, mixer: str, torch):
    try:
        from raven.models.raven import RavenForCausalLM
    except ModuleNotFoundError as exc:
        if exc.name == "fla":
            raise SystemExit(
                "Missing `flash-linear-attention` (`fla`). Install Raven dependencies in this Python environment, "
                "for example: `pip install -e /home/xiaol/X/raven`."
            ) from exc
        raise

    dtype = _torch_dtype(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config = _build_config(args, mixer)
    model = RavenForCausalLM(config).to(device=device, dtype=dtype)
    model.train(args.backward)

    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone() if args.backward else None
    total_steps = args.warmup + args.steps
    losses = []

    _sync(torch, args.device)
    start = None
    for step in range(total_steps):
        if step == args.warmup:
            _sync(torch, args.device)
            start = time.perf_counter()

        if args.backward:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss
            loss.backward()
            losses.append(float(loss.detach().float().cpu()))
        else:
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
                losses.append(float(out.logits.detach().float().mean().cpu()))

    _sync(torch, args.device)
    elapsed = time.perf_counter() - start if start is not None else 0.0
    tokens = args.batch_size * args.seq_len * args.steps
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0

    return {
        "mixer": mixer,
        "params_m": _param_count(model) / 1e6,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "backward": args.backward,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
        "peak_memory_gb": peak_gb,
        "last_metric": losses[-1] if losses else None,
    }


def _print_results(results: list[dict]) -> None:
    print("\n| mixer | params M | tok/s | peak GB | seconds | metric |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            f"| {row['mixer']} | {row['params_m']:.2f} | {row['tokens_per_second']:.0f} | "
            f"{row['peak_memory_gb']:.2f} | {row['seconds']:.3f} | {row['last_metric']:.4f} |"
        )


def _jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = _parse_args()
    _validate_cuda_rwkv_args(args)
    _maybe_prepend_sys_path(_repo_root())
    _configure_lt2_paths(args)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required. Install Raven's dependencies first, for example `pip install -e .`.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    results = []
    for mixer in args.mixers:
        print(f"Benchmarking {mixer}...")
        results.append(_bench_one(args, mixer, torch))

    _print_results(results)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"args": _jsonable_args(args), "results": results}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
