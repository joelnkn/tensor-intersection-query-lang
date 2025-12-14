import torch


torch._dynamo.config.capture_dynamic_output_shape_ops = True


def hand_intersect(query, A, B, device=None):
    # assert query == "A[i] == B[j]"
    sorted_values, indices = torch.sort(A)
    keys = B
    search = torch.searchsorted(sorted_values, keys)
    mask = sorted_values[search % len(sorted_values)] == keys

    dim1 = indices[search[mask]].unsqueeze(1)
    dim2 = mask.nonzero()

    ind = torch.cat((dim1, dim2), dim=1)
    return ind


def hand_reduce(query, A, B, device=None):
    # A[i,c] == B[j,c] -> (i,j)
    all_vectors = torch.cat([A, B], dim=0)
    _, vector_labels = torch.unique(all_vectors, dim=0, return_inverse=True)
    values = vector_labels[: A.shape[0]]
    keys = vector_labels[B.shape[0] :]

    sorted_values, indices = torch.sort(values)
    search = torch.searchsorted(sorted_values, keys)
    mask = sorted_values[search % len(sorted_values)] == keys

    dim1 = indices[search[mask]].unsqueeze(1)
    dim2 = mask.nonzero()

    ind = torch.cat((dim1, dim2), dim=1)
    return ind


def hand_shared_intersect(query, A, B, device=None):
    # A[i,j] == B[j,k]
    # Flatten
    m, n = A.shape
    p = B.shape[1]

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)

    # Build indices for A: k -> (k // n, k % n)
    kA = torch.arange(m * n, device=device, dtype=torch.int64)
    A_i = torch.div(kA, n, rounding_mode="floor")
    A_j = kA - A_i * n

    # Build indices for B: k -> (k // p, k % p)
    kB = torch.arange(n * p, device=device, dtype=torch.int64)
    B_j = torch.div(kB, p, rounding_mode="floor")
    B_k = kB - B_j * p

    vec0 = torch.stack([A_j, A_flat], dim=1)
    vec1 = torch.stack([B_j, B_flat], dim=1)

    ind = hand_reduce("A[i,c] == B[j,c] -> (i,j)", vec0, vec1, device)

    return torch.stack(
        [
            A_i[ind[:, 0]],
            A_j[ind[:, 0]],
            B_k[ind[:, 0]],
        ],
        dim=1,
    )


def pack_u32_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: int32 or int64 tensors with values in [0, 2^32)
    returns: uint64 tensor where key = (a << 32) | b
    """
    a = a.to(torch.int64)
    b = b.to(torch.int64)
    return (a << 32) | b


def hand_shared_intersect32(query, A, B, device=None):
    # A[i,j] == B[j,k]
    # Flatten
    m, n = A.shape
    p = B.shape[1]

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)

    # Build indices for A: k -> (k // n, k % n)
    kA = torch.arange(m * n, device=device, dtype=torch.int64)
    A_i = torch.div(kA, n, rounding_mode="floor")
    A_j = kA - A_i * n

    # Build indices for B: k -> (k // p, k % p)
    kB = torch.arange(n * p, device=device, dtype=torch.int64)
    B_j = torch.div(kB, p, rounding_mode="floor")
    B_k = kB - B_j * p

    pack0 = pack_u32_pair(A_j, A_flat)
    pack1 = pack_u32_pair(B_j, B_flat)

    ind = hand_intersect("A[i] == B[j] -> (i,j)", pack0, pack1, device)

    return torch.stack(
        [
            A_i[ind[:, 0]],
            A_j[ind[:, 0]],
            B_k[ind[:, 0]],
        ],
        dim=1,
    )


def hand_table(query, A, B, device=None):
    return torch.nonzero(A[:, None] == B[None, :])


if __name__ == "__main__":
    # print(hand_intersect("fjak", torch.tensor([1, 2, 3]), torch.tensor([3, 2, 4])))
    # print(hand_table("fjak", torch.tensor([1, 2, 3]), torch.tensor([3, 2, 4])))
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Default CUDA device
        print("Using CUDA:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # test_case = {
    #     "hand_kernel": hand_intersect,
    #     "query": "A[i] == B[j] -> (i,j)",
    #     "tensor_dims": {"A": (10,), "B": (10,)},
    # }
    test_case = {
        "hand_kernel": hand_shared_intersect,
        "query": "A[i, j] == B[j, k] -> (i,j,k)",
        "tensor_dims": {"A": (10, 10), "B": (10, 10)},
    }

    tensors = {
        name: torch.randint(size=dims, high=10, device=device)
        for name, dims in test_case["tensor_dims"].items()
    }

    result = test_case["hand_kernel"](test_case["query"], **tensors, device=device)

    # with torch._inductor.utils.fresh_inductor_cache():
    #     kernel = test_case["hand_kernel"]
    #     compiled_intersect = torch.compile(kernel, dynamic=True)
    #     # compiled_intersect = table_intersect
    #     result = compiled_intersect(test_case["query"], **tensors, device=device)

# TODO: special case 1, no shared indices, 2, A[i] == B[j]
# TODO: read on prefix-sum
