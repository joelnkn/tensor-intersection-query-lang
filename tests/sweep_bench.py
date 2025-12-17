# sweep_bench.py
import argparse
import itertools
from pathlib import Path

from tests.benchmark import QUERY_SPECS, QueryParams, benchmark_query

# Usage: python -m tests.sweep_bench --out results.json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="JSONL output path")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)

    # which queries (by index into QUERY_SPECS)
    p.add_argument(
        "--queries",
        type=int,
        nargs="+",
        default=None,
        help="comma-separated indices like 0,2,3 or 'all'",
    )

    # sweep ranges (small + sane defaults)
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2**12, 2**14],
        help="comma-separated sizes",
    )
    p.add_argument(
        "--share-ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Share ratios",
    )

    p.add_argument(
        "--skews",
        type=float,
        nargs="+",
        default=[0.2, 1.0, 5.0],
        help="Skew factors",
    )

    p.add_argument(
        "--reduce-dims",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Reduction dimensions",
    )

    # method toggles
    p.add_argument("--no-flags", action="store_true")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-hand", action="store_true")

    return p.parse_args()


def parse_list(s, cast=float):
    return s


def main():
    args = parse_args()

    sizes = parse_list(args.sizes, int)
    share_ratios = parse_list(args.share_ratios, float)
    skews = parse_list(args.skews, float)
    reduce_dims = parse_list(args.reduce_dims, int)

    if args.queries is None:
        q_indices = list(range(len(QUERY_SPECS)))
    else:
        q_indices = args.queries

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_idx = 0
    for qi in q_indices:
        query, shape_factory, hand_kernel = QUERY_SPECS[qi]
        if args.no_hand:
            hand_kernel = None

        # Only sweep reduce_dim for queries whose shape_factory uses it.
        # Cheap heuristic: just sweep it always; it won't matter for factories that ignore it.
        for size, skew, share_ratio, reduce_dim in itertools.product(
            sizes, skews, share_ratios, reduce_dims
        ):
            params = QueryParams.from_args(
                size=size,
                args=argparse.Namespace(  # emulate your argparse behavior
                    share_ratio=share_ratio,
                    skew=skew,
                    reduce_dim=reduce_dim,
                ),
            )

            benchmark_query(
                query=query,
                shape_factory=shape_factory,
                params=params,
                warmup=args.warmup,
                trials=args.trials,
                hand_kernel=hand_kernel,
                no_flags=args.no_flags,
                no_table=args.no_table,
                index=run_idx,
                out_jsonl=str(out_path),
                method_tag="sweep_v1",
            )
            run_idx += 1


if __name__ == "__main__":
    main()
