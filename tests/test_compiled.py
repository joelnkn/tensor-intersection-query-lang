import pytest
import math
from tiql import table_intersect
import torch


torch._dynamo.config.capture_dynamic_output_shape_ops = True
device = None


@pytest.mark.parametrize(
    "query, tensor_dims",
    [
        ("A[i] == B[j]", {"A": (10,), "B": (10,)}),
        # ("A[i, j] == B[j]", {"A": (10, 10), "B": (10,)}),
        # ("A[i, c] == B[j, c] -> (i,c)", {"A": (10, 2), "B": (10, 2)}),
        # ("A[i, c] == B[j, c] -> (i)", {"A": (10, 2), "B": (10, 2)}),
        # ("A[i, j] == B[k]", {"A": (10, 20), "B": (10,)}),
        # ("A[i, j] == B[j, k] -> (i,j,k)", {"A": (10, 15), "B": (15, 8)}),
        # ("A[i, j] == B[j, k] -> (i,k)", {"A": (10, 5), "B": (5, 10)}),
    ],
)
def test_compiled_output(query, tensor_dims):
    # tensors = {
    #     name: torch.randint(size=dims, high=10, device=device)
    #     for name, dims in tensor_dims.items()
    # }
    tensors = generate_random_unique_tensors(tensor_dims)

    with torch._inductor.utils.fresh_inductor_cache():
        compiled_intersect = torch.compile(table_intersect, dynamic=True)
        result = compiled_intersect(query, **tensors, device=device).to(
            dtype=torch.bool
        )

    expected = table_intersect(query, **tensors, device=device)
    assert torch.all(expected == result)


def generate_random_unique_tensors(
    shape_spec, *, max_value=50, device="cpu", dtype=torch.long
):
    """
    For each tensor:
      - values are random unique integers within that tensor
      - values come from [0, max_value)
      - different tensors can share values
    """
    tensors = {}

    for name, shape in shape_spec.items():
        numel = math.prod(shape)

        if max_value < numel:
            raise ValueError(
                f"max_value={max_value} must be >= tensor size {numel} "
                f"to ensure uniqueness."
            )

        # Sample a unique random subset of size numel, then reshape
        vals = torch.randperm(max_value, device=device, dtype=dtype)[:numel]
        tensors[name] = vals.reshape(shape)

    return tensors
