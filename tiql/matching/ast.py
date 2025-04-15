from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Set
from numbers import Number
import torch
from .range import Range, DataRange


# ===================
# Base Node
# ===================


@dataclass
class ASTNode:
    def run(self, device: torch.Device, data: dict, idx_range: Range | DataRange):
        raise NotImplementedError()


# ===================
# Expressions
# ===================


@dataclass
class Constant(ASTNode):
    value: Union[int, float]

    def run(self, device: torch.Device, data: dict, idx_range: Range):
        return self.value


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: str  # "+", "-", "*"
    right: ASTNode

    def run(self, device: torch.Device, data: dict, idx_range: Range) -> DataRange:
        left_data: DataRange | Number = self.left.run(device, data, idx_range)
        right_data: DataRange | Number = self.right.run(device, data, idx_range)
        match self.op:
            case "+":
                return left_data.cross_add(right_data)
            case "-":
                return left_data.cross_sub(right_data)
            case "*":
                return left_data.cross_mul(right_data)


@dataclass
class FuncCall(ASTNode):
    func: str  # "min" or "max"
    args: List[ASTNode]  # exactly two arguments

    def run(self, device: torch.Device, data: dict, idx_range: Range) -> DataRange:
        raise NotImplementedError()


@dataclass
class Access(ASTNode):
    tensor: str
    indices: List[str | Access]  # positions can be identifiers or nested accesses

    def run(self, device: torch.Device, data: dict, idx_range: Range) -> DataRange:
        indices = [
            idx if isinstance(idx, str) else idx.run(device, data, idx_range)
            for idx in self.indices
        ]
        return DataRange.from_tensor(data[self.tensor], indices, device, idx_range)


# ===================
# Query Expressions
# ===================


@dataclass
class QueryExpr(ASTNode):
    left: ASTNode
    op: str  # "==", ">=", "<=", "<", ">"
    right: ASTNode

    def run(self, device: torch.Device, data: dict, idx_range: Range) -> Range:
        left_data: DataRange | Number = self.left.run(device, data, idx_range)
        right_data: DataRange | Number = self.right.run(device, data, idx_range)

        # print(f"Expr inputs: \n{left_data}\n\n{self.op}\n\n{right_data}\n\n")
        match self.op:
            case "==":
                return left_data.intersect_eq(right_data)
            case ">=":
                return left_data.intersect_ge(right_data)
            case "<=":
                return left_data.intersect_le(right_data)
            case "<":
                return left_data.intersect_lt(right_data)
            case ">":
                return left_data.intersect_gt(right_data)


# ===================
# Full Query (comma-separated q_expr list)
# ===================


@dataclass
class Query(ASTNode):
    expressions: List[QueryExpr]
    out_indices: tuple[str] = field(default_factory=tuple)

    def run(self, device: torch.Device, data: dict, idx_range: Range = None) -> Range:
        out = Range.empty(device)
        for expr in self.expressions:
            out = expr.run(device, data, out)

        return out.to_tensor()
