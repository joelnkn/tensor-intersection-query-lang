import torch
import sys
from tiql.solver import Solver


def generate_random_tensors():
    # Shapes are chosen to support a variety of queries
    return {
        "A": torch.randint(0, 10, (3, 3)),
        "B": torch.randint(0, 10, (3,)),
        "C": torch.randint(0, 10, (3, 3)),
        "D": torch.randint(0, 10, (3,)),
    }


def print_tensor_data(tensor_data):
    print("Generated Tensors:")
    for name, tensor in tensor_data.items():
        print(f"{name}:\n{tensor}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py '<query>'")
        return

    query = sys.argv[1]
    tensor_data = generate_random_tensors()
    print_tensor_data(tensor_data)

    solver = Solver()
    try:
        result = solver.solve(query, tensor_data)
        print(f"Result for query '{query}':\n{result}")
    except Exception as e:
        print(f"Error solving query: {e}\n\n")
        raise e


if __name__ == "__main__":
    main()
