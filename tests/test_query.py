import pytest
import torch
from tiql.solver import Solver
from tiql.matching import Range

simple_tensor_data = {
    "A": torch.tensor([[1, 2], [3, 4]]),  # Shape (2,2)
    "B": torch.tensor([1, 3]),  # Shape (2,)
    "C": torch.tensor([[2, 5], [4, 7]]),  # Shape (2,2)
}

solver = Solver()

# A[i] = B[j] -> (i,j)
# A[i] + B[k] = C[j] -> (i,k,j)
# A[i,c] + B[k,c] = C[j,c] -> (i,j,k)
# Max(AL[i], BL[j]) <= Min(AR[i], BR[j])


@pytest.mark.parametrize(
    "query, expected_output",
    [
        ("A[i, j] == B[j]", torch.tensor([[0], [0]])),
        ("A[i,j] == B[k]", torch.tensor([[1, 0], [0, 0], [1, 0]])),
        ("A[i, j] + B[j] == C[i, j]", torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])),
        ("A[i, j] + B[j] == C[k, i]", torch.tensor([[0, 1], [0, 1], [0, 1]])),
        ("A[i,j] < C[i,j]", torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])),
        ("A[i,j] > C[i,j]", torch.tensor([[], []])),
    ],
)
def test_simple_eq_query(query, expected_output):
    result = solver.solve(query, simple_tensor_data)
    assert_equal_output(result, expected_output, query)


def assert_equal_output(result: torch.Tensor, expected: torch.Tensor, query: str):
    unique_result = result.unique(dim=1)
    assert (
        unique_result.shape[1] == result.shape[1]
    ), "All outputed index columns must be unique"

    unique_expected = expected.unique(dim=1)
    assert (
        unique_result.shape[1] == result.shape[1]
    ), "All expected index columns must be unique"

    unique_both = torch.cat([unique_expected, unique_result], dim=1).unique(dim=1)
    assert (
        unique_result.shape[1] == unique_expected.shape[1] == unique_both.shape[1]
    ), f"Failed for query: {query}\nExpected:\n{expected}\n\nGot:\n{result}\n"
