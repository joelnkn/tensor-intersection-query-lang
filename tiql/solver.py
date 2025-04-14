import typing
import torch
from tiql.parsing import parse
from tiql.matching import Range


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
            print("Using CUDA:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU.")

    def solve(self, query: str, data: dict) -> Range:
        """
        Solves a query using the configured device.

        Args:
            query: The input query to solve.
        """
        print(f"Solving query on device: {self.device}")
        query_ast = parse(query)
        solution = query_ast.run(self.device, data)
        return solution
