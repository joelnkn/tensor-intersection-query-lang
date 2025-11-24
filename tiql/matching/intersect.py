import torch


def _sort_segment_intersect(
    v: torch.Tensor, w: torch.Tensor, device, v_mask=None, w_mask=None
):
    if len(v) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # TODO: only perform if dimension unsorted
    v_sorted, v_indices = torch.sort(v)

    _, v_counts = torch.unique_consecutive(v_sorted, return_counts=True)
    # O(unique(W) log V + W log W + V log V)
    search = torch.searchsorted(v_sorted, w)
    search_clamped = torch.clamp(search, 0, len(v_sorted) - 1)

    mask = (search < len(v_sorted)) & (v_sorted[search_clamped] == w)
    if v_mask:
        mask &= v_mask[v_indices[search_clamped]]
    if w_mask:
        mask &= w_mask

    v_intersect = search[
        mask
    ]  # v_intersect contains the intersection indices for v_unique
    counts = v_counts.repeat_interleave(v_counts)[v_intersect]

    increment = torch.arange(torch.sum(counts), device=device)
    offset = torch.repeat_interleave(counts.cumsum(dim=0) - counts, counts)

    dim1 = v_indices[v_intersect.repeat_interleave(counts) + increment - offset]
    dim2 = mask.nonzero().repeat_interleave(counts)

    return torch.stack((dim1, dim2), dim=1)


def _chained_intersect(
    tensors: list[torch.Tensor], masks: list[torch.Tensor | None], device
):
    """Output is shape (k,n) where k = len(tensors)"""
    assert len(tensors) == len(masks)

    if len(tensors) == 1:
        if masks[0] is None:
            return torch.arange(len(tensors[0]), device=device).unsqueeze(1)

        return torch.arange(len(tensors[0]), device=device)[masks[0]].unsqueeze(1)

    inter = _sort_segment_intersect(
        tensors[0], tensors[1], device=device, v_mask=masks[0], w_mask=masks[1]
    )
    for tensor, mask in zip(tensors[2:], masks[2:]):
        values = tensors[0][inter[:, 0]]
        nex = _sort_segment_intersect(values, tensor, device=device, w_mask=mask)
        inter = torch.cat((inter[nex[:, 0], :], nex[:, 1:]), dim=1)
    return inter


intersect = _sort_segment_intersect
chained_intersect = _chained_intersect
