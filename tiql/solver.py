from typing import Optional
import torch
from tiql.parsing import parse
from tiql.matching import Range

import tiql.compiler_passes.compiler_passes


def intersect(query: str, device: Optional[torch.device] = None, **kwargs):
    """
    Solves a tensor index query of the form 'A[i] + B[j] == C[k]' and returns all index
    tuples that satisfy the expression.

    Args:
        query (str): The query string to solve, e.g., "A[i] + B[j] == C[k]".
        device (torch.device, optional): The device to perform computation on
            (e.g., torch.device('cuda') or torch.device('cpu')). If not provided,
            CUDA will be used if available, otherwise CPU.
        **kwargs: Named tensors used in the query (e.g., A=tensorA, B=tensorB, C=tensorC).

    Returns:
        torch.Tensor: A tensor of shape (D, N) where each column is a D-dimensional tuple
        of indices (e.g., (i, j, k)) that satisfy the query.
    """
    solver = Solver(device=device)
    return solver.solve(query, kwargs)


def table_intersect(
    query: str, device: Optional[torch.device] = None, **kwargs
) -> torch.tensor:
    """Intersect using an intersection table backend"""
    solver = Solver(device=device)
    return solver.solve(query, kwargs, table=True)


def simple_intersect(
    query: str, device: Optional[torch.device] = None, **kwargs
) -> torch.tensor:
    """Intersect simple queries. All input tensors must be single dimensional,
    with simple indices (no nesting). Only equality operator allowed. Can have multiple queries.

    eg.
    A[i] == B[j] == C[i], A[j] == C[k]
    """
    solver = Solver(device=device)
    return solver.solve_simple(query, kwargs)


class Solver:
    device: torch.device

    def __init__(self, device: torch.device = None):
        """
        Initializes the solver with the specified device.

        Args:
            device (str, optional): The device to use ("cpu" or "cuda").
            Defaults to CUDA if available.
        """
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # Default CUDA device
            # print("Using CUDA:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            # print("CUDA not available, using CPU.")

    def solve(self, query: str, data: dict, table: bool = False) -> torch.tensor:
        """
        Solves a query using the configured device.

        Args:
            query: The input query to solve.
        """
        # print(f"Solving query on device: {self.device}")
        query_ast = parse(query)
        # table = True
        if table:
            solution = query_ast.table_run(self.device, data)
        else:
            solution = query_ast.run(self.device, data)

        return solution

    def solve_simple(self, query: str, data: dict) -> torch.tensor:
        query_ast = parse(query)
        return query_ast.simple_run(self.device, data)
