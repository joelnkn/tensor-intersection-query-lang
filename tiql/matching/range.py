from __future__ import annotations
from numbers import Number
from typing import Sequence
import torch
from .intersect import intersect


class Range:
    """
    Stores a range of indices with symbols for use in matching.
    """

    indices: torch.Tensor  # (k,n) tensor for k dimensional range
    symbols: dict[str, int]  # maps each symbol to its corresponding row in `indices`
    device: torch.Device

    def __init__(
        self, indices: torch.Tensor, symbols: dict[str, int], device: torch.Device
    ):
        self.indices = indices
        self.symbols = symbols
        self.device = device

    @classmethod
    def empty(cls, device: torch.Device) -> Range:
        return cls(torch.tensor(device=device), {}, device)

    @classmethod
    def from_shape(
        cls, shape: torch.Size, symbols: Sequence[str], device: torch.Device
    ) -> Range:
        assert len(shape) == len(
            symbols
        ), "Number of symbols should match number of dimensions."

        indices_per_dim = [torch.arange(dim, device) for dim in shape]

        # Create meshgrid and stack to get index tuples
        grid = torch.meshgrid(*indices_per_dim, indexing="ij")
        stacked = torch.stack(grid, dim=0)  # Shape [k, d1, d2, ..., dk]

        # Reshape to [k, numel] and convert to same dtype as original tensor
        indices = stacked.reshape(len(shape), -1)  # k x n shape
        return cls(indices, symbols, device)

    def index_intersect(self, other: Range, shared: set = None) -> torch.tensor:
        """
        Returns a n by 2 tensor where rows consist of all (i,j) such that
        `self.indices[i]` matches `other.indices[j]` between all shared symbols.
        If `shared` is provided, only those symbols are considered.
        If the tensors share no symbols, all pairs of index pairs are returned.
        """
        if shared is None:
            shared = self.symbols.keys() & other.symbols.keys()

        if len(shared) > 1:
            self_symbols, other_symbols = [], []
            for i in shared:
                self_symbols.append(self.symbols[i])
                other_symbols.append(other.symbols[i])

            joined_indices = torch.cat(
                (self.indices[self_symbols, :], other.indices[other_symbols, :]), dim=1
            )
            _, index_labels = torch.unique(
                joined_indices, dim=1, sorted=True, return_inverse=True
            )
            self_canon_idx = index_labels[: self.indices.shape[1]]
            other_canon_idx = index_labels[-other.indices.shape[1] :]

            return intersect(self_canon_idx, other_canon_idx, self.device)

        elif len(shared) == 1:
            return intersect(
                self.indices[[self.symbols[0]], :].squeeze(0),
                other.indices[[other.symbols[0]], :].squeeze(0),
                self.device,
            )

        else:
            # complete pairwise cross product of `self` and `other`
            n = self.indices.shape[1]
            m = other.indices.shape[1]

            i_vals = torch.arange(n).repeat_interleave(m)
            j_vals = torch.arange(m).repeat(n)

            return torch.stack((i_vals, j_vals), dim=1)

    def join(self, other: Range, join_idx: torch.Tensor, shared: set = None) -> Range:
        """
        Returns a new Range by joining this Range with `other`, combining indices and symbols
        according to `join_idx`.

        Each row in `join_idx` is a pair `(i, j)` indicating that `self.indices[i]`
        should be joined with `other.indices[j]`. Thus, `join_idx` is of shape (n, 2).

        Wherever symbols are shared between `self` and `other`, indices from `self` are used.
        """
        if shared is None:
            shared = self.symbols.keys() & other.symbols.keys()

        if len(shared) >= 1:
            other_unshared = other.symbols.keys() - shared

            if other_unshared:
                symbols = self.symbols.copy()

                # reassign all symbols exclusively in `other` to appear after
                # those in `self`. ensure correct order for future select.
                other_symbols = []
                for i, symb in enumerate(other_unshared):
                    symbols[symb] = i + len(self.symbols)
                    other_symbols.append(other.symbols(symb))

                # select only indices in `other` that aren't shared with `self`
                other_indices = other.indices[
                    torch.tensor(other_symbols, device=self.device), join_idx[:, 1]
                ]

                # stack the two disjoint sets of indices
                indices = torch.cat(
                    (self.indices[:, join_idx[:, 0]], other_indices), dim=0
                )

                return Range(indices, symbols, self.device)

            else:
                return Range(
                    self.indices[:, join_idx[:, 0]], self.symbols.copy(), self.device
                )

        else:
            symbols = {
                **self.symbols,
                **{k: v + len(self.symbols) for k, v in other.symbols.items()},
            }

            return Range(
                torch.cat(
                    (self.indices[:, join_idx[:, 0]], other.indices[:, join_idx[:, 1]]),
                    dim=0,
                ),
                symbols,
                self.device,
            )

    def cross(self, other: Range) -> Range:
        """
        Returns a Range consisting of (self.indices[i] | other.indices[j]) for all pairs (i,j)
        where the shared indices between self.indices[i] and other.indices[j] are exactly the same.
        """
        shared = self.symbols.keys() & other.symbols.keys()
        int_idx = self.index_intersect(other, shared)
        return self.join(other, int_idx, shared)


class DataRange:
    """
    Stores a range of indices with symbols for use in matching, with corresponding data entries
    for each index vector.
    """

    index_range: Range
    data: torch.Tensor

    def __init__(
        self,
        index_range: Range,
        data: torch.Tensor,
    ):
        assert index_range.indices.shape[1] == data.shape[0]
        self.index_range = index_range
        self.data = data

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        ordered_symbols: Sequence[str | DataRange],
        device: torch.Device,
    ):
        # Get the shape and number of elements
        shape = tensor.shape  # Tuple of dimensions
        numel = tensor.numel()  # Total number of elements

        # Generate indices using meshgrid, then reshape to (k, numel)
        indices = torch.stack(
            torch.meshgrid(
                *(torch.arange(dim, dtype=torch.int64) for dim in shape),
                indexing="ij",
            ),
            dim=0,
        ).reshape(
            len(shape), numel
        )  # Shape: (k, numel)

        # Flatten the data tensor
        data = tensor.flatten()  # Shape: (numel,)
        symbols = {symb: i for i, symb in enumerate(ordered_symbols)}

        return DataRange(
            Range(indices, symbols, device),
            data,
        )

    def _intersect_cmp(self, other: DataRange | Number, comparison_fn) -> Range:
        # TODO: add special case code for when shared == 0.
        # Here, we should intersect over data, not indices.

        int_idx = self.index_range.index_intersect(other.index_range)
        mask = comparison_fn(self.data[int_idx[:, 0]], other.data[int_idx[:, 1]])
        eq_idx = int_idx[mask]
        return self.index_range.join(other.index_range, eq_idx)

    def intersect_eq(self, other: DataRange | Number) -> Range:
        return self._intersect_cmp(other, torch.eq)

    def intersect_lt(self, other: DataRange | Number):
        return self._intersect_cmp(other, torch.lt)

    def intersect_le(self, other: DataRange | Number):
        return self._intersect_cmp(other, torch.le)

    def intersect_gt(self, other: DataRange | Number):
        return self._intersect_cmp(other, torch.gt)

    def intersect_ge(self, other: DataRange | Number):
        return self._intersect_cmp(other, torch.ge)

    def _cross_arithmetic(self, other: DataRange | Number, operation_fn) -> DataRange:
        int_idx = self.index_range.index_intersect(other.index_range)
        data = operation_fn(self.data[int_idx[:, 0]], other.data[int_idx[:, 1]])
        new_range = self.index_range.join(
            other.index_range,
            int_idx,
        )
        return DataRange(new_range, data)

    def cross_add(self, other: DataRange | Number) -> DataRange:
        return self._cross_arithmetic(other, torch.add)

    def cross_sub(self, other: DataRange | Number) -> DataRange:
        return self._cross_arithmetic(other, torch.sub)

    def cross_mul(self, other: DataRange | Number) -> DataRange:
        return self._cross_arithmetic(other, torch.mul)
