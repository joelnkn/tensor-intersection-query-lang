import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import statistics
import time
from typing import Callable, Dict, Iterable, Tuple

from tiql.handwritten import (
    hand_intersect,
    hand_reduce,
    hand_shared_intersect,
    hand_shared_intersect32,
)
from tiql.compiler_passes.compiler_passes import update_all_compiler_passes
from tiql.matching import Query
import torch

from tiql import table_intersect
import logging

logger = logging.getLogger(__name__)


torch._dynamo.config.capture_dynamic_output_shape_ops = True
if torch.cuda.is_available():
    device = torch.device("cuda")  # Default CUDA device
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

ShapeFactory = Callable[[int], Dict[str, Tuple[int, ...]]]
QuerySpec = Tuple[str, ShapeFactory, Callable]


@dataclass(frozen=True)
class QueryParams:
    size: int
    share_ratio: float = 0.5
    skew: float = 1.0
    reduce_dim: int = 4

    @classmethod
    def from_args(
        cls,
        *,
        size: int,
        args,
    ) -> "QueryParams":
        """
        Build QueryParams from an argparse.Namespace.
        Any field set to None in args falls back to the class default.
        """

        def pick(name):
            val = getattr(args, name, None)
            return val if val is not None else getattr(cls, name)

        return cls(
            size=size,
            share_ratio=pick("share_ratio"),
            skew=pick("skew"),
            reduce_dim=pick("reduce_dim"),
        )


def _dims_Ai_eq_Bj(p: QueryParams) -> Dict[str, Tuple[int, ...]]:
    return {"A": (int(p.size * p.skew),), "B": (int(p.size / p.skew),)}


def _dims_Aic_eq_Bjc(p: QueryParams) -> Dict[str, Tuple[int, ...]]:
    return {
        "A": (int(p.size * p.skew / p.reduce_dim), p.reduce_dim),
        "B": (int(p.size / (p.reduce_dim * p.skew)), p.reduce_dim),
    }


def _dims_Aij_eq_Bjk(p: QueryParams) -> Dict[str, Tuple[int, ...]]:
    return {
        "A": (
            int(((p.size * p.skew)) ** (1 - p.share_ratio)),
            int((p.size) ** (p.share_ratio)),
        ),
        "B": (
            int((p.size) ** (p.share_ratio)),
            int((p.size / p.skew) ** (1 - p.share_ratio)),
        ),
    }


def _dims_Ai_eq_Bjk(p: QueryParams) -> Dict[str, Tuple[int, ...]]:
    side = int((p.size / p.skew) ** 0.5)
    return {"A": (int(p.size * p.skew),), "B": (side, side)}


QUERY_SPECS: Tuple["QuerySpec", ...] = (
    (
        "A[i] == B[j]",
        _dims_Ai_eq_Bj,
        hand_intersect,
    ),
    (
        "A[i, c] == B[j, c] -> (i,j)",
        _dims_Aic_eq_Bjc,
        hand_reduce,
    ),
    (
        "A[i, j] == B[j, k] -> (i,j,k)",
        _dims_Aij_eq_Bjk,
        hand_shared_intersect32,
    ),
    (
        "A[i] == B[j, k] -> (i,j,k)",
        _dims_Ai_eq_Bjk,
        None,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eager vs compiled kernel runtimes."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        help="Kernel sizes (N) used to build tensor shapes. "
        "Pass multiple values to parameterize the benchmark.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of timed runs used for each measurement.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Exclude custom compiler flags from benchmark",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Exclude table intersection from benchmark",
    )
    parser.add_argument("-s", "--skew", type=float, help="Update skew")
    parser.add_argument("-r", "--share-ratio", type=float, help="Update share ratio")
    parser.add_argument(
        "-c", "--reduce-dim", type=int, help="Update reduce dim size (c)"
    )
    parser.add_argument(
        "-q", "--query", nargs="+", type=int, help="Choose queries to run"
    )
    return parser.parse_args()


def generate_random_unique_tensors(shape_spec, *, max_value=50, dtype=torch.long):
    """
    Equivalent helper from tests/test_compiled.py, but exposed locally so the
    benchmark can scale tensor shapes with the requested kernel size.
    """
    tensors = {}

    for name, shape in shape_spec.items():
        numel = math.prod(shape)
        if max_value < numel:
            raise ValueError(
                f"max_value={max_value} must be >= tensor size {numel} "
                f"to ensure uniqueness."
            )
        vals = torch.randperm(max_value, device=device, dtype=dtype)[:numel]
        tensors[name] = vals.reshape(shape)

    return tensors


def time_callable(fn, *, warmup: int, trials: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()

    timings = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    mean = statistics.mean(timings)
    stdev = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return mean, stdev


def measure_peak_cuda_bytes(fn, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    return peak, reserved_peak


def benchmark_query(
    query: str,
    shape_factory: ShapeFactory,
    params: QueryParams,
    *,
    warmup: int,
    trials: int,
    hand_kernel: Callable | None = None,
    no_flags: bool = False,
    no_table: bool = False,
    index: int = None,
    return_record: bool = False,
    out_jsonl: str | None = None,
    method_tag: str | None = None,
) -> None:
    shape_spec = shape_factory(params)
    max_tensor_size = max(math.prod(shape) for shape in shape_spec.values())
    tensors = generate_random_unique_tensors(
        shape_spec,
        max_value=max(2 * max_tensor_size, 50),
    )

    # initialize torch compile. get rid of big print block
    torch.compile(lambda x: x + 1)(torch.zeros((1,), device=device))

    header = (
        f"\n\n{index}: [query={query!r} size={params.size} skew={params.skew} "
        f"share_ratio={params.share_ratio} reduce_dim={params.reduce_dim}]"
    )
    print(header)

    record = {
        "ts": time.time(),
        "index": index,
        "query": query,
        "method_tag": method_tag,
        "params": (
            asdict(params)
            if hasattr(params, "__dataclass_fields__")
            else {
                "size": params.size,
                "share_ratio": params.share_ratio,
                "skew": params.skew,
                "reduce_dim": params.reduce_dim,
            }
        ),
        "shape_spec": {k: list(v) for k, v in shape_spec.items()},
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": {},  # filled below
    }

    eager_time = eager_std = None
    compiled_time = compiled_std = None
    compiled_with_flags_time = compiled_with_flags_std = None
    hand_time = hand_std = None

    if not no_table:
        eager = lambda: table_intersect(query, **tensors, device=device)
        with torch._inductor.utils.fresh_inductor_cache():
            torch._dynamo.reset()
            update_all_compiler_passes(False)
            eager_time, eager_std = time_callable(eager, warmup=warmup, trials=trials)

            compiled = torch.compile(table_intersect, dynamic=True)
            compiled(query, **tensors, device=device)
            compiled_call = lambda: compiled(query, **tensors, device=device)
            compiled_time, compiled_std = time_callable(
                compiled_call, warmup=warmup, trials=trials
            )

        print(
            f"eager          = {eager_time:.6f}s±{eager_std:.6f}s "
            f"\ncompiled (tab) = {compiled_time:.6f}s±{compiled_std:.6f}s "
        )
        record["results"]["eager_table"] = {"mean_s": eager_time, "std_s": eager_std}
        record["results"]["compiled_table"] = {
            "mean_s": compiled_time,
            "std_s": compiled_std,
        }

    if not no_flags:
        with torch._inductor.utils.fresh_inductor_cache():
            torch._dynamo.reset()
            update_all_compiler_passes(True)
            compiled_with_flags = torch.compile(table_intersect, dynamic=True)
            compiled_with_flags(query, **tensors, device=device)
            compiled_with_flags_call = lambda: compiled_with_flags(
                query, **tensors, device=device
            )

            compiled_with_flags_time, compiled_with_flags_std = time_callable(
                compiled_with_flags_call, warmup=warmup, trials=trials
            )

        print(
            f"compiled (bin) = {compiled_with_flags_time:.6f}s±{compiled_with_flags_std:.6f}s "
        )
        record["results"]["compiled_flags"] = {
            "mean_s": compiled_with_flags_time,
            "std_s": compiled_with_flags_std,
        }

    if hand_kernel:
        with torch._inductor.utils.fresh_inductor_cache():
            hand_eager = lambda: hand_kernel(query, **tensors, device=device)
            hand_time, hand_std = time_callable(
                hand_eager, warmup=warmup, trials=trials
            )

        print(f"handwritten    = {hand_time:.6f}s±{hand_std:.6f}s ")
        record["results"]["handwritten"] = {"mean_s": hand_time, "std_s": hand_std}

    if not no_flags and not no_table and compiled_with_flags_time:
        speedup = eager_time / compiled_with_flags_time
        print(f"speedup        = {speedup:.2f}x")
        record["results"]["speedup_eager_over_flags"] = speedup

    if out_jsonl is not None:
        os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record if return_record else None


def main() -> None:
    args = parse_args()

    for i, (query, shape_factory, hand_kernel) in enumerate(QUERY_SPECS):
        if args.query is not None and i not in args.query:
            continue

        for size in args.sizes:
            params = QueryParams.from_args(size=size, args=args)
            benchmark_query(
                query,
                shape_factory,
                params,
                warmup=args.warmup,
                trials=args.trials,
                hand_kernel=hand_kernel,
                no_flags=args.no_flags,
                no_table=args.no_table,
                index=i,
            )


if __name__ == "__main__":
    main()
