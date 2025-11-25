import math
from typing import Mapping
import torch
from torch._inductor.pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    filter_nodes,
    fwd_only,
    get_arg_value,
    get_mutation_region_id,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MultiOutputPattern,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)
from torch._inductor.fx_passes.post_grad import pass_patterns

aten = torch.ops.aten

# Compiler pass flags - set to False to disable a pass
# COMPILER_PASS_FLAGS = {
#     "pointless_bwand_replacement": False,
#     "replace_table_intersection": False,
#     "remove_scatter_nonzero": False,
#     "replace_intersect_and": False,
#     "replace_reduce_all": False,
# }

COMPILER_PASS_FLAGS = {
    "pointless_bwand_replacement": True,
    "replace_table_intersection": True,
    "remove_scatter_nonzero": True,
    "replace_intersect_and": True,
    "replace_reduce_all": True,
}


def _log_flags(prefix: str) -> None:
    print(f"{prefix}\n{COMPILER_PASS_FLAGS}")


def get_compiler_pass_flags() -> dict:
    """Return a copy of the current compiler pass flag map."""
    return COMPILER_PASS_FLAGS.copy()


def set_compiler_pass_flag(flag: str, *, enabled: bool) -> None:
    """Set an individual compiler pass flag at runtime."""
    if flag not in COMPILER_PASS_FLAGS:
        raise KeyError(
            f"Unknown compiler pass flag '{flag}'. "
            f"Available flags: {sorted(COMPILER_PASS_FLAGS)}"
        )
    COMPILER_PASS_FLAGS[flag] = bool(enabled)
    _log_flags(f"Updated compiler flag '{flag}'")


def update_all_compiler_passes(enabled: bool) -> None:
    for flag in COMPILER_PASS_FLAGS:
        COMPILER_PASS_FLAGS[flag] = enabled

    _log_flags(f"Updated all flags to '{enabled}'")


def update_compiler_pass_flags(overrides: Mapping[str, bool]) -> None:
    """Bulk update compiler pass flags."""
    unknown = set(overrides).difference(COMPILER_PASS_FLAGS)
    if unknown:
        raise KeyError(
            f"Unknown compiler pass flag(s) {sorted(unknown)}. "
            f"Available flags: {sorted(COMPILER_PASS_FLAGS)}"
        )
    for flag, enabled in overrides.items():
        COMPILER_PASS_FLAGS[flag] = bool(enabled)
    _log_flags("Updated compiler flags")


_log_flags("Compiling with flags")


def check_can_replace_pointless_bwand(match):
    if not COMPILER_PASS_FLAGS["pointless_bwand_replacement"]:
        return False
    fill_value = match.kwargs["fill_value"]
    return fill_value


@register_graph_pattern(
    CallFunction(
        aten.bitwise_and.Tensor,
        CallFunction(
            aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            # _users=MULTIPLE,
        ),
        KeywordArg("t1"),
        # _users=MULTIPLE,
    ),
    pass_dict=pass_patterns[1],
    extra_check=check_can_replace_pointless_bwand,
)
@register_graph_pattern(
    CallFunction(
        aten.bitwise_and.Tensor,
        KeywordArg("t1"),
        CallFunction(
            aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            # _users=MULTIPLE,
        ),
        # _users=MULTIPLE,
    ),
    pass_dict=pass_patterns[1],
    extra_check=check_can_replace_pointless_bwand,
)
def pointless_bwand_replacement(match: Match, shape, fill_value, device, dtype, t1):
    def repl(x):
        return x

    # only replace the output node, not all nodes
    match.replace_by_example(repl, [t1])


def check_can_replace_with_bin_search(match):
    s0 = match.kwargs["t0"].meta["val"].shape
    s1 = match.kwargs["t1"].meta["val"].shape

    # positions of non-1 index is different
    if len(s0) != len(s1) or len(s0) != 2:
        return False

    int0 = None
    int1 = None
    for i, (d0, d1) in enumerate(zip(s0, s1)):
        if d0 == 1 and d1 > 1:
            if int1 is None:
                int1 = i
            else:
                return False

        if d0 > 1 and d1 == 1:
            if int0 is None:
                int0 = i
            else:
                return False

    match.int0 = int0
    match.int1 = int1
    return int0 is not None and int1 is not None


# def unravel_tensor(shape: tuple[int], device):
#     numel = math.prod(shape)
#     indices = torch.arange(numel, dtype=torch.long, device=device)
#     return torch.stack(torch.unravel_index(indices, shape), dim=0)


def unravel_tensor(shape: tuple[int, ...], device=None):
    if device is None:
        device = torch.device("cpu")
    numel = math.prod(shape)
    flat = torch.arange(numel, dtype=torch.long, device=device)

    # Python-int strides
    strides = [math.prod(shape[d + 1 :]) for d in range(len(shape))]
    coords = [(flat // s) % dim for s, dim in zip(strides, shape)]
    return torch.stack(coords, dim=0)


def check_can_replace_table_intersection(match):
    if not COMPILER_PASS_FLAGS["replace_table_intersection"]:
        return False
    s0 = match.kwargs["t0"].meta["val"].shape
    s1 = match.kwargs["t1"].meta["val"].shape

    inc0 = {i for i, sz in enumerate(s0) if sz > 1}
    inc1 = {i for i, sz in enumerate(s1) if sz > 1}

    # intersect over shared dimensions + data
    shared = inc0 & inc1
    match.shared = shared

    return len(inc0 - shared) and len(inc1 - shared)


# A[i] == B[j,k]
# A[i,k] == B[j, k]  -> stack data with k recipe


def vector_search_intersect(value_vectors: torch.Tensor, key_vectors: torch.Tensor):
    """Find intersection indices as in search_intersect, however intersect is performed
    with vectors as values.

    Args:
        sorted_vectors (torch.Tensor): shape: [n, c]
        key_vectors (torch.Tensor): shape: [m, c]
    """
    joined_vectors = torch.cat((value_vectors, key_vectors), dim=0)
    _, vector_labels = torch.unique(joined_vectors, dim=0, return_inverse=True)
    values = vector_labels[: value_vectors.shape[0]]
    keys = torch.flip(
        torch.flip(vector_labels, dims=[0])[: key_vectors.shape[0]], dims=[0]
    )

    # sorted_values = values
    sorted_values, indices = torch.sort(values)
    search = torch.searchsorted(sorted_values, keys)
    sorted_values = torch.cat(
        (sorted_values, torch.full((1,), fill_value=-1, device=value_vectors.device))
    )
    mask = sorted_values[search] == keys

    dim1 = indices[search[mask]].unsqueeze(1)
    dim2 = mask.nonzero()

    return dim1, dim2


def search_intersect_unique(sorted_values: torch.Tensor, keys: torch.Tensor):
    """
    Find intersection indicies between the sorted_values tensor (which is sorted in ascending
    order) and the keys tensor. Both tensors have no duplicates and are 1-dimensional.
    """

    # k: [1, 3, 4, 7, 9]
    # v: [0, 1, 2, 3 ... 7]

    # search: [1, 3, 4, 7, 8]
    #

    search = torch.searchsorted(sorted_values, keys)
    mask = sorted_values[search % len(sorted_values)] == keys

    dim1 = search[mask].unsqueeze(1)
    dim2 = mask.nonzero()

    return dim1, dim2


# if value is not sorted:
#     ...
#     torch.sort(v)
#     ...
# if value is sorted and unique:
#     int_idx = torch.searchsorted()
#     out = torch.zeros(shape)
#     out[int_idx] = 1
# elif value is unsorted and unique:
#     sort_value, sort_idx = torch.sort(value)
#     int_idx = torch.searchsorted()
#     out = torch.zeros(shape)
#     out[sort_idx[int_idx]] = 1
@register_graph_pattern(
    CallFunction(
        aten.eq.Tensor,
        KeywordArg("t0"),
        KeywordArg("t1"),
    ),
    pass_dict=pass_patterns[1],
    extra_check=check_can_replace_table_intersection,
)
def replace_table_intersection(match: Match, t0, t1):
    # print("found bin search opp", t0, t1)

    # def repl_basic(v0, v1, int0, int1):
    #     sorted_values = v0.squeeze()
    #     keys = v1.squeeze()

    #     dim1, dim2 = search_intersect_unique(sorted_values, keys)

    #     size = (v0.shape[int0], v1.shape[int1])

    #     out = torch.zeros(size, dtype=torch.long)
    #     out = out.index_put((dim1, dim2), torch.ones((1, 1)))
    #     return out

    # return match.replace_by_example(
    #     repl_basic,
    #     [t0, t1, 1, 0],
    # )

    def repl(t0, t1, shared):
        A0 = unravel_tensor(t0.shape, t0.device)
        A1 = unravel_tensor(t1.shape, t0.device)

        data0 = torch.flatten(t0)
        data1 = torch.flatten(t1)

        ind0 = [A0[i] for i in shared]
        ind1 = [A1[i] for i in shared]

        vec0 = torch.stack(ind0 + [data0], dim=1)
        vec1 = torch.stack(ind1 + [data1], dim=1)

        fdim0, fdim1 = vector_search_intersect(vec0, vec1)
        dim0 = A0[:, fdim0]
        dim1 = A1[:, fdim1]

        z = torch.zeros((1,), device=t0.device, dtype=torch.long)
        d = []
        for i, (d0, d1) in enumerate(zip(t0.shape, t1.shape)):
            if d0 > 1:
                d.append(dim0[i])
            elif d1 > 1:
                d.append(dim1[i])
            else:
                d.append(z)

        size = tuple(max(sz0, sz1) for sz0, sz1 in zip(t0.shape, t1.shape))

        # TODO: instead of using `d` here, place 1s into flat tensor and then reshape to size
        out = torch.zeros(size, dtype=torch.long)
        out = out.index_put(d, torch.ones((1,), device=t0.device))
        return out

    return match.replace_by_example(
        repl,
        [t0, t1, match.shared],
    )


"""
unsqueeze_default: "i64[u5, 1][1, 1]cpu" = torch.ops.aten.unsqueeze.default(index_tensor_1, 1);  index_tensor_1 = None
nonzero_default: "i64[u5, 1][1, u5]cpu" = torch.ops.aten.nonzero.default(eq_tensor);  eq_tensor = None
full_default_6: "i64[100, 100][100, 1]cpu" = torch.ops.aten.full.default([100, 100], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
full_default_7: "f32[1, 1][1, 1]cpu" = torch.ops.aten.full.default([1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
index_put_default: "i64[100, 100][100, 1]cpu" = torch.ops.aten.index_put_.default(full_default_6, [unsqueeze_default, nonzero_default], full_default_7);  full_default_6 = unsqueeze_default = nonzero_default = full_default_7 = None

 # File: /Users/joelmanu/Documents/speinsum/speinsum/compiler.py:458 in sparse_einsum, code: int_idx = torch.nonzero(table_run).T
nonzero: "i64[u0, 2][1, u0]cpu" = torch.ops.aten.nonzero.default(index_put_default);
"""


def check_can_replace_scatter_nonzero(match: Match):
    if not COMPILER_PASS_FLAGS["remove_scatter_nonzero"]:
        return False
    return match.kwargs["fill_val"] == 0


# NOTE: technically wrong if this is filled with more zeros. should add extra check in future
# for correctness
@register_graph_pattern(
    CallFunction(
        aten.nonzero.default,
        CallFunction(
            aten.index_put.default,
            CallFunction(
                aten.full.default,
                Arg(),
                KeywordArg("fill_val"),
            ),
            KeywordArg("indices"),
            Arg(),
        ),
    ),
    pass_dict=pass_patterns[2],
    extra_check=check_can_replace_scatter_nonzero,
)
def remove_scatter_nonzero(match: Match, *args, fill_val, indices):
    def repl(indices):
        return torch.cat(indices, dim=1)

    return match.replace_by_example(
        repl,
        [indices],
    )


def check_can_repace_intersect_and(match: Match):
    if not COMPILER_PASS_FLAGS["replace_intersect_and"]:
        return False
    # ind0 = match.kwargs["ind0"][0].meta["val"]
    # ind1 = match.kwargs["ind1"][0].meta["val"]
    # new_size = ind0.size(0) + ind1.size(0)
    # torch._constrain_as_size(ind0.size(0), min=2, max=1000)
    # torch._constrain_as_size(ind1.size(0), min=2, max=1000)

    # torch.fx.experimental.symbolic_shapes.guard_size_oblivious(new_size >= 2)

    # joined_vectors = torch.cat((value_vectors, key_vectors), dim=0)
    # joined_vectors = torch.zeros((new_size,))
    # return False
    return all(
        (
            match.kwargs["shape0"] == match.kwargs["shape1"],
            match.kwargs["fill0"] == match.kwargs["fill1"],
            len(match.kwargs["ind0"]) == len(match.kwargs["ind1"]),
        )
    )


# this one should match the index-scatter as well, as it otherwise replaces
# all uses of bitwise_and


# TODO: handle broadcasting correctly
@register_graph_pattern(
    CallFunction(
        aten.bitwise_and.Tensor,
        CallFunction(
            aten.index_put.default,
            CallFunction(
                aten.full.default,
                KeywordArg("shape0"),
                KeywordArg("fill0"),
            ),
            KeywordArg("ind0"),
            Arg(),
        ),
        CallFunction(
            aten.index_put.default,
            CallFunction(
                aten.full.default,
                KeywordArg("shape1"),
                KeywordArg("fill1"),
            ),
            KeywordArg("ind1"),
            Arg(),
        ),
    ),
    pass_dict=pass_patterns[2],
    extra_check=check_can_repace_intersect_and,
)
def replace_intersect_and(match: Match, *args, **kwargs):
    def repl(size, ind0, ind1):
        ind0 = torch.cat(ind0, dim=1)
        ind1 = torch.cat(ind1, dim=1)

        # THESE ARE UNIQUE!!
        shared, _ = vector_search_intersect(ind0, ind1)
        dim = tuple(ind0[shared, i] for i in range(len(size)))

        out = torch.zeros(size, dtype=torch.long)
        out = out.index_put(dim, torch.ones((1,)))

        return out

    return match.replace_by_example(
        repl,
        [kwargs["shape0"], kwargs["ind0"], kwargs["ind1"]],
    )


def check_can_replace_reduce_all(match: Match):
    if not COMPILER_PASS_FLAGS["replace_reduce_all"]:
        return False
    return match.kwargs["fill"] == 0


@register_graph_pattern(
    CallFunction(
        aten.logical_not.default,
        CallFunction(
            aten.any.dims,
            CallFunction(
                aten.logical_not.default,
                CallFunction(
                    aten.index_put.default,
                    CallFunction(
                        aten.full.default,
                        KeywordArg("shape"),
                        KeywordArg("fill"),
                    ),
                    KeywordArg("ind"),
                    Arg(),
                ),
            ),
            KeywordArg("dims"),
        ),
    ),
    pass_dict=pass_patterns[2],
    extra_check=check_can_replace_reduce_all,
)
def replace_reduce_all(match: Match, *args, **kwargs):
    def repl(ind, reduce_dims, shape):
        reduce_sizes = (s for i, s in enumerate(shape) if i in reduce_dims)
        k = math.prod(reduce_sizes)

        relevant = [v for i, v in enumerate(ind) if i not in reduce_dims]
        relevant_ind = torch.cat(relevant, dim=1)

        unique_indices, counts = torch.unique(relevant_ind, dim=0, return_counts=True)
        reduced_ind = unique_indices[counts == k, :]
        size = [s for i, s in enumerate(shape) if i not in reduce_dims]
        dim = tuple(reduced_ind[:, i] for i in range(len(size)))

        out = torch.zeros(size, dtype=torch.long)
        out = out.index_put(dim, torch.ones((1,)))

        return out

    return match.replace_by_example(
        repl,
        [kwargs["ind"], kwargs["dims"], kwargs["shape"]],
    )


# TODO:

# imp tested
# ✅    support reduction
# ✅    achieve correctness on duplicates
#       ideal intersection kernel A[i,k] == B[k,j]
#       investigate search_sorted sorter=
