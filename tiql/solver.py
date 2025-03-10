import typing
import torch
from tiql.parsing import parse
import tiql.matching as matching


class Solver:
    device: torch.Device

    def __init__(self, device: torch.Device = None):
        """
        Initializes the solver with the specified device.

        Args:
            device (str, optional): The device to use ("cpu" or "cuda").
            Defaults to CUDA if available.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Default CUDA device
            print("Using CUDA:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU.")

    def solve(self, query: str, data: dict) -> None:
        """
        Solves a query using the configured device.

        Args:
            query: The input query to solve.
        """
        print(f"Solving query on device: {self.device}")
        query_ast = parse(query)
        solution = query_ast.run()
