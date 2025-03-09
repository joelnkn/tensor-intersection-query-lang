from dataclasses import dataclass
from typing import List, Union


# ===================
# Base Node
# ===================


@dataclass
class ASTNode:
    def run(self) -> None:
        raise NotImplementedError()


# ===================
# Expressions
# ===================


@dataclass
class Identifier(ASTNode):
    name: str


@dataclass
class Constant(ASTNode):
    value: Union[int, float]


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: str  # "+", "-", "*"
    right: ASTNode


@dataclass
class FuncCall(ASTNode):
    func: str  # "min" or "max"
    args: List[ASTNode]  # exactly two arguments


@dataclass
class Access(ASTNode):
    tensor: str
    indices: List[ASTNode]  # positions can be identifiers or nested accesses


# ===================
# Query Expressions
# ===================


@dataclass
class QueryExpr(ASTNode):
    left: ASTNode
    op: str  # "==", ">=", "<=", "<", ">"
    right: ASTNode


# ===================
# Full Query (comma-separated q_expr list)
# ===================


@dataclass
class Query(ASTNode):
    expressions: List[QueryExpr]
