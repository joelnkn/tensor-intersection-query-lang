import torch


def sort_segment_intersect(v: torch.Tensor, w: torch.Tensor, device):
    if len(v) == 0:
        return torch.empty((0, 2), dtype=torch.long)
    v_sorted, v_indices = torch.sort(v)

    _, v_counts = torch.unique_consecutive(v_sorted, return_counts=True)
    search = torch.searchsorted(v_sorted, w)

    w_mask = (search < len(v_sorted)) & (
        v_sorted[torch.clamp(search, 0, len(v_sorted) - 1)] == w
    )

    v_intersect = search[
        w_mask
    ]  # v_intersect contains the intersection indices for v_unique
    counts = v_counts.repeat_interleave(v_counts)[v_intersect]

    increment = torch.arange(torch.sum(counts), device=device)
    offset = torch.repeat_interleave(counts.cumsum(dim=0) - counts, counts)

    dim1 = v_indices[v_intersect.repeat_interleave(counts) + increment - offset]
    dim2 = w_mask.nonzero().repeat_interleave(counts)

    return torch.stack((dim1, dim2), dim=1)


intersect = sort_segment_intersect
