from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union
from numbers import Number
import torch
from .range import Range, DataRange
from .indirecteinsum import einsum_gs


# ===================
# Base Node
# ===================


@dataclass
class ASTNode:
    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range | DataRange,
    ):
        raise NotImplementedError()

    def table_run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_order: tuple[str],
    ) -> torch.Tensor:
        raise NotImplementedError()


# ===================
# Expressions
# ===================


@dataclass
class Constant(ASTNode):
    value: Union[int, float]

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range,
    ):
        return self.value


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: str  # "+", "-", "*"
    right: ASTNode

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range,
    ) -> DataRange:
        left_data: DataRange | Number = self.left.run(
            device, data, out_indices, idx_range
        )
        right_data: DataRange | Number = self.right.run(
            device, data, out_indices, idx_range
        )
        match self.op:
            case "+":
                return left_data.cross_add(right_data)
            case "-":
                return left_data.cross_sub(right_data)
            case "*":
                return left_data.cross_mul(right_data)

    def table_run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_order: tuple[str],
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: ...
        """
        left_table: torch.Tensor = self.left.table_run(
            device, data, out_indices, idx_order
        )
        right_table: torch.Tensor = self.right.table_run(
            device, data, out_indices, idx_order
        )
        match self.op:
            case "+":
                return left_table + right_table
            case "-":
                return left_table - right_table
            case "*":
                return left_table * right_table


@dataclass
class FuncCall(ASTNode):
    func: str  # "min" or "max"
    args: List[ASTNode]  # exactly two arguments

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range,
    ) -> DataRange:
        raise NotImplementedError()


@dataclass
class Access(ASTNode):
    tensor: str
    indices: List[str | Access]  # positions can be identifiers or nested accesses

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range,
    ) -> DataRange:
        indices = [
            (
                idx
                if isinstance(idx, str)
                else idx.run(device, data, out_indices, idx_range)
            )
            for idx in self.indices
        ]
        return DataRange.from_tensor(data[self.tensor], indices, device, idx_range)

    def table_run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_order: tuple[str],
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: ...
        """

        # A[i, k] == B[j, k]   ->  L(I, J, K)  == (I, J, K)

        # L[i,k] = A[i, i, B[k]]
        # L.uns(1) => L

        tensor_shape = data[self.tensor].shape
        out = torch.zeros(
            tuple(
                tensor_shape[self.indices.index(idx)]
                for idx in idx_order
                if idx in self.indices
            ),
            dtype=torch.long,
        )
        gather_query = f"Out[{','.join([self.format_index(idx, idx_order) for idx in idx_order if idx in self.indices])}] = {self.text(idx_order)}"
        out = einsum_gs(
            gather_query,
            Out=out,
            **data,
        )

        # einsum_query = f"{''.join([self.format_index(idx, idx_order) for idx in idx_order if idx in self.indices])}->"

        for i, idx in enumerate(idx_order):
            if idx not in self.indices:
                out = out.unsqueeze(i)

        return out

    def text(self, idx_order) -> str:
        indices = [
            (
                self.format_index(idx, idx_order)
                if isinstance(idx, str)
                else idx.text(idx_order)
            )
            for idx in self.indices
        ]
        return f"{self.tensor}[{','.join(indices)}]"

    def format_index(self, idx: str, idx_order: tuple[str]):
        return chr(idx_order.index(idx) + 97)


# ===================
# Query Expressions
# ===================


@dataclass
class QueryExpr(ASTNode):
    left: ASTNode
    op: str | None  # "==", ">=", "<=", "<", ">"
    right: ASTNode | None

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_range: Range,
    ) -> Range:
        left_data: DataRange | Number = self.left.run(
            device, data, out_indices, idx_range
        )

        if self.op is None:
            return left_data.index_range

        right_data: DataRange | Number = self.right.run(
            device, data, out_indices, idx_range
        )

        # print(f"Expr inputs: \n{left_data}\n\n{self.op}\n\n{right_data}\n\n")
        match self.op:
            case "==":
                return left_data.intersect_eq(right_data, out_indices)
            case ">=":
                return left_data.intersect_ge(right_data, out_indices)
            case "<=":
                return left_data.intersect_le(right_data, out_indices)
            case "<":
                return left_data.intersect_lt(right_data, out_indices)
            case ">":
                return left_data.intersect_gt(right_data, out_indices)

    def table_run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str],
        idx_order: tuple[str],
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: ...
        """
        left_data: torch.Tensor = self.left.table_run(
            device, data, out_indices, idx_order
        )

        if self.op is None:
            return torch.ones(left_data.shape, dtype=torch.bool), False

        right_data: torch.Tensor = self.right.table_run(
            device, data, out_indices, idx_order
        )

        match self.op:
            case "==":
                return left_data == right_data, True
            case ">=":
                return left_data >= right_data, True
            case "<=":
                return left_data <= right_data, True
            case "<":
                return left_data < right_data, True
            case ">":
                return left_data > right_data, True


# ===================
# Full Query (comma-separated q_expr list)
# ===================


@dataclass
class Query(ASTNode):
    expressions: List[QueryExpr]
    out_indices: tuple[str] = None
    idx_order: tuple[str] = None  # TODO: this should never be done

    def run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str] = None,
        idx_range: Range = None,
    ) -> Range:
        out = Range.empty(device) if idx_range is None else idx_range
        for expr in self.expressions:
            out = expr.run(device, data, self.out_indices, out)

        return out.to_tensor(self.out_indices)

    def table_run(
        self,
        device: torch.Device,
        data: dict,
        out_indices: tuple[str] = None,
        idx_order: tuple[str] = None,
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: ...
        """
        table = None

        # track whether any dynamic operators where actually used,
        # to determine if nonzero has to be called
        dynamic = False
        for expr in self.expressions:
            run_table, run_dynamic = expr.table_run(
                device, data, out_indices, self.idx_order
            )
            dynamic = dynamic or run_dynamic

            if table is None:
                table = run_table
            else:
                table = table & run_table

        # return table, dynamic

        if dynamic:
            return torch.nonzero(table).T
            # return torch.ones((len(table.shape), 5), dtype=torch.long)
        else:
            return torch.stack(
                torch.meshgrid(
                    *[torch.arange(s, device=table.device) for s in table.shape],
                    indexing="ij",
                ),
                dim=0,
            ).reshape(len(table.shape), -1)
