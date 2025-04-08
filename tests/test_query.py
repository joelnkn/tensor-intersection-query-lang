import pytest
import torch
from tiql.solver import Solver
from tiql.matching import Range

tensor_data = {
    "A": torch.tensor([[1, 2], [3, 4]]),  # Shape (2,2)
    "B": torch.tensor([1, 3]),  # Shape (2,)
    "C": torch.tensor([[2, 5], [4, 7]]),  # Shape (2,2)
}

solver = Solver()

# A[i] = B[j] -> (i,j)
# A[i] + B[k] = C[j] -> (i,k,j)
# A[i,c] + B[k,c] = C[j,c] -> (i,j,k)
# Max(AL[i], BL[j]) <= Min(AR[i], BR[j])


def test_simple_query():
    query = "A[i,j] == B[k]"
    result = solver.solve(query, tensor_data)
    assert result == Range.from_indices(
        [(1, 0, 1), (0, 0, 0)], ("i", "j", "k"), solver.device
    ), str(result)


@pytest.mark.parametrize(
    "query, expected_output",
    [
        ("A[i] == B[j]", [(0, 0), (1, 1)]),
        ("A[i, j] + B[j] == C[i, j]", [(0, 0), (1, 1)]),
        ("A[i, j] * B[j] == C[i, j]", [(0, 1), (1, 0)]),
        ("A[i, j] + B[j] == C[k, i], A[j, k] == C[i, j]", [(0, 0, 1)]),
        ("A[i, j] - B[j] == C[i, j]", [(1, 0)]),
    ],
)
def test_tensor_query(query, expected_output):
    result = solver.solve(query, tensor_data)
    print(result)
    # assert result == expected_output, f"Failed for query: {query}"
