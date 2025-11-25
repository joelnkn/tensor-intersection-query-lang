import argparse
import math
import statistics
import time
from typing import Callable, Dict, Iterable, Tuple

from tiql.compiler_passes.compiler_passes import update_all_compiler_passes
import torch

from tiql import table_intersect


torch._dynamo.config.capture_dynamic_output_shape_ops = True
if torch.cuda.is_available():
    device = torch.device("cuda")  # Default CUDA device
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

ShapeFactory = Callable[[int], Dict[str, Tuple[int, ...]]]
QuerySpec = Tuple[str, ShapeFactory]


QUERY_SPECS: Tuple[QuerySpec, ...] = (
    ("A[i] == B[j]", lambda size: {"A": (size,), "B": (size,)}),
    (
        "A[i, c] == B[j, c] -> (i,j)",
        lambda size: {"A": (size, 4), "B": (size, 4)},
    ),
    (
        "A[i, c] == B[j, c] -> (i)",
        lambda size: {"A": (size, 4), "B": (size, 4)},
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
        help="Use no custom compiler flags for test",
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


def benchmark_query(
    query: str,
    shape_factory: ShapeFactory,
    sizes: Iterable[int],
    *,
    warmup: int,
    trials: int,
) -> None:
    for size in sizes:
        shape_spec = shape_factory(size)
        max_tensor_size = max(math.prod(shape) for shape in shape_spec.values())
        tensors = generate_random_unique_tensors(
            shape_spec,
            max_value=max(2 * max_tensor_size, 50),
        )

        eager = lambda: table_intersect(query, **tensors, device=device)

        # with torch._inductor.utils.fresh_inductor_cache():
        #     update_all_compiler_passes(False)
        #     compiled = torch.compile(table_intersect, dynamic=True)
        #     compiled(query, **tensors, device=device)
        #     compiled_call = lambda: compiled(query, **tensors, device=device)

        with torch._inductor.utils.fresh_inductor_cache():
            compiled_with_flags = torch.compile(table_intersect, dynamic=True)
            compiled_with_flags(query, **tensors, device=device)
            compiled_with_flags_call = lambda: compiled_with_flags(
                query, **tensors, device=device
            )

            eager_time, eager_std = time_callable(eager, warmup=warmup, trials=trials)
            # compiled_time, compiled_std = time_callable(
            #     compiled_call, warmup=warmup, trials=trials
            # )
            compiled_with_flags_time, compiled_with_flags_std = time_callable(
                compiled_with_flags_call, warmup=warmup, trials=trials
            )

        speedup = (
            eager_time / compiled_with_flags_time
            if compiled_with_flags_time
            else float("inf")
        )
        print(
            f"\n\n[query={query!r} size={size}] "
            f"\neager={eager_time:.6f}s±{eager_std:.6f}s "
            # f"\ncompiled={compiled_time:.6f}s±{compiled_std:.6f}s "
            f"\ncompiled={compiled_with_flags_time:.6f}s±{compiled_with_flags_std:.6f}s "
            f"\nspeedup={speedup:.2f}x"
        )


def main() -> None:
    args = parse_args()
    update_all_compiler_passes(not args.no_flags)
    for query, shape_factory in QUERY_SPECS:
        benchmark_query(
            query,
            shape_factory,
            args.sizes,
            warmup=args.warmup,
            trials=args.trials,
        )


if __name__ == "__main__":
    main()
