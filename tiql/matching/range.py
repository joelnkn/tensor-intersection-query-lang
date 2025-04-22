from __future__ import annotations
from numbers import Number
from typing import Sequence, Tuple
import torch
from .intersect import intersect


class Range:
    """
    Stores a range of indices with symbols for use in matching.
    """

    indices: torch.Tensor  # (k,n) tensor for k dimensional range
    symbols: dict[str, int]  # maps each symbol to its corresponding row in `indices`
    # size: dict[str, int]  # maps each symbol to the size of its corresponding dimension
    device: torch.Device

    def __init__(
        self, indices: torch.Tensor, symbols: dict[str, int], device: torch.Device
    ):
        self.indices = indices
        self.symbols = symbols
        self.device = device

    @classmethod
    def empty(cls, device: torch.Device) -> Range:
        return cls(torch.empty((0, 1), dtype=torch.int, device=device), {}, device)

    @classmethod
    def from_indices(
        cls,
        indices: Sequence[Tuple[str, ...]],
        symbols: Tuple[str, ...],
        device: torch.Device,
    ) -> Range:
        new_symbols = {symb: i for i, symb in enumerate(symbols)}
        new_indices = torch.tensor(indices).T
        return cls(new_indices, new_symbols, device)

    @classmethod
    def from_shape(
        cls, shape: torch.Size, symbols: Sequence[str], device: torch.Device
    ) -> Range:
        assert len(shape) == len(
            symbols
        ), "Number of symbols should match number of dimensions."

        indices_per_dim = [torch.arange(dim, device=device) for dim in shape]

        # Create meshgrid and stack to get index tuples
        grid = torch.meshgrid(*indices_per_dim, indexing="ij")
        stacked = torch.stack(grid, dim=0)  # Shape [k, d1, d2, ..., dk]

        # Reshape to [k, numel] and convert to same dtype as original tensor
        indices = stacked.reshape(len(shape), -1)  # k x n shape
        return cls(indices, symbols, device)

    def __eq__(self, other):
        if not isinstance(other, Range):
            return False

        # Ensure devices match
        if self.device != other.device:
            return False

        # Ensure symbol mappings are the same
        if self.symbols != other.symbols:
            return False

        # Extract rows in order dictated by `symbols`
        key_order = list(self.symbols.keys())
        self_ordered = self.indices[[self.symbols[key] for key in key_order], :]
        # Selects rows in order of symbols
        other_ordered = other.indices[[other.symbols[key] for key in key_order], :]

        # Check if they contain the same set of index tuples (ignore order)
        self_unique = torch.unique(self_ordered, dim=1)
        other_unique = torch.unique(other_ordered, dim=1)

        return torch.equal(self_unique, other_unique)

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
            shared_index = next(iter(shared))
            return intersect(
                self.indices[self.symbols[shared_index], :],
                other.indices[other.symbols[shared_index], :],
                self.device,
            )

        else:
            # TODO: investigate torch.cartesion_prod
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
                other_unshared_indices = []
                for i, symb in enumerate(other_unshared):
                    symbols[symb] = i + len(self.symbols)
                    other_unshared_indices.append(other.symbols[symb])

                # select only indices in `other` that aren't shared with `self`
                other_indices = other.indices[other_unshared_indices][:, join_idx[:, 1]]

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

    def get(self, symbol: str) -> torch.Tensor:
        """
        Returns the row corresponding to all entries of a particular symbol stored in this Range,
        in the same order that it is stored.
        """
        return self.indices[self.symbols[symbol], :]

    def to_tensor(self, symbol_order: tuple[str]):
        """
        Given an ordering of symbols of length k, return a k by n tensor with rows corresponding the
        index dimensions in the specified order. If no order is given, the all symbols will be used
        in alphabetical order.
        """
        symbols = [self.symbols[symb] for symb in symbol_order]
        return self.indices[symbols, :]

    def __str__(self):
        return f"{self.symbols}\n{self.indices}"


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
        assert (
            index_range.indices.shape[1] == data.shape[0]
        ), f"{index_range.indices.shape} vs {data.shape}"
        self.index_range = index_range
        self.data = data

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        index: Sequence[str | DataRange],
        device: torch.Device,
        index_range: Range = None,
    ):
        # TODO: should take and use index_range as an input (possibly None)

        # index_range stores all tuples of input indices into the query that correspond to a
        # specifc data value.
        if index_range is None:
            index_range = Range.empty(device)

        # all the values needed to index into the real tensor, built iteratively
        data_indices = torch.empty(
            (0, index_range.indices.shape[1]), dtype=torch.int, device=device
        )

        for i, idx in enumerate(index):
            if isinstance(idx, DataRange):
                join_idx = index_range.index_intersect(idx.index_range)

                index_range = index_range.join(idx.index_range, join_idx)
                data_indices = torch.cat(
                    (data_indices[:, join_idx[:, 0]], idx.data[:, join_idx[:, 1]]),
                    dim=0,
                )
            else:
                assert isinstance(idx, str)

                if idx in index_range.symbols:
                    # TODO clamp range of idx given dimension size
                    symbol_indices = index_range.get(idx).unsqueeze(0)
                    data_indices = torch.cat((data_indices, symbol_indices), dim=0)

                else:
                    new_dim = Range.from_shape((tensor.size(i),), {idx: 0}, device)
                    join_idx = index_range.index_intersect(new_dim)

                    index_range = index_range.join(new_dim, join_idx)
                    data_indices = torch.cat(
                        (
                            data_indices[:, join_idx[:, 0]],
                            new_dim.indices[:, join_idx[:, 1]],
                        ),
                        dim=0,
                    )

        data = tensor[tuple(data_indices)]

        return DataRange(
            index_range,
            data,
        )

    def _intersect_cmp(
        self, other: DataRange | Number, comparison_fn, out_indices: tuple[str]
    ) -> Range:
        # TODO: add special case code for when shared == 0.
        # Here, we should intersect over data, not indices.
        # why don't we stack data and indices and run intersect directly on that?

        # Note, there are 3 options.
        # Do index intersect first, then data comparison (easy)
        # Do data comparison first, then index intersect (easy)
        # Do index and data comparisons at the same time.
        #   Need general logical intersect (eg. all i,j pairs v[i] < w[j])
        #   Stack indices and data (I, D) and run at most two logical intersects.
        #   eg. say we have A = I_1, D_1, B = I_2, D_2, find all index tuples i,j A[i] <= B[j]
        #   do (I_1, D_1) <= (I_2, D_2) and (I_2, -D_2) <= (I_1, -D_1), intersect the results
        #

        int_idx = self.index_range.index_intersect(other.index_range)
        mask = comparison_fn(self.data[int_idx[:, 0]], other.data[int_idx[:, 1]])
        eq_idx = int_idx[mask]
        result = self.index_range.join(other.index_range, eq_idx)

        if out_indices is not None:
            for symb in self.index_range.symbols:
                if symb not in out_indices:
                    k = 3  # Size along the reduction dimension

                    num_indices = result.indices.shape[0]
                    exclude_idx = result.symbols[symb]
                    new_symbols = {
                        k: (v - 1 if v > exclude_idx else v)
                        for k, v in result.symbols.items()
                        if k != symb
                    }
                    keep_indices = [j for j in range(num_indices) if j != exclude_idx]

                    # Sort lexicographically (torch.unique implicitly does this)
                    unique_indices, counts = torch.unique(
                        result.indices[keep_indices, :],
                        dim=1,
                        return_counts=True,
                    )

                    # Check that every unique group appears exactly `k` times
                    result.indices = unique_indices[:, counts == k]
                    result.symbols = new_symbols

        return result

    def intersect_eq(self, other: DataRange | Number, out_indices: tuple[str]) -> Range:
        return self._intersect_cmp(other, torch.eq, out_indices)

    def intersect_lt(self, other: DataRange | Number, out_indices: tuple[str]) -> Range:
        return self._intersect_cmp(other, torch.lt, out_indices)

    def intersect_le(self, other: DataRange | Number, out_indices: tuple[str]) -> Range:
        return self._intersect_cmp(other, torch.le, out_indices)

    def intersect_gt(self, other: DataRange | Number, out_indices: tuple[str]) -> Range:
        return self._intersect_cmp(other, torch.gt, out_indices)

    def intersect_ge(self, other: DataRange | Number, out_indices: tuple[str]) -> Range:
        return self._intersect_cmp(other, torch.ge, out_indices)

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

    def __str__(self):
        return f"{self.index_range}\n{self.data}"
