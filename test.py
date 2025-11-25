from tiql.solver import intersect, table_intersect
import torch
import torch._inductor.utils

# torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# TORCH_LOGS_FORMAT=“%(levelname)s:%(message)s” TORCH_LOGS="aot_graphs" python test.py
# TORCH_LOGS_FORMAT=“%(levelname)s:%(message)s” TORCH_LOGS="aot_graphs, post_grad_graphs" python test.py
# TORCH_COMPILE_DEBUG=1 python test.py

# Data
if torch.cuda.is_available():
    device = torch.device("cuda")  # Default CUDA device
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

# Currently : Table (o), Binarysearch (o)
test_case = {
    "query": "A[i] == B[j] -> (i,j)",
    "tensor_dims": {"A": (10,), "B": (10,)},
}

# Currently : Table + torch.all (o)
# TODO : Reduction Compiler pass
# Expected: single binary search -> torch.unique on c dimension
#   1. coo only with c + data (Acoo, Bcoo)
#   2. Binary search Acoo vs. Bcoo
#

# A = torch.randn((128,3))
# B = torch.randn((256,3))

# Acoo = torch.stack([torch.arange(128*3)%3, A.flatten()]) # 128*3, 2
# Bcoo = torch.stack([torch.arange(256*3)%3, B.flatten()]) # 256*3, 2
# Acoo, Bcoo = torch.unique(Acoo, Bcoo) # 128*3, 256*3
# Bcoo = torch.sort(Bcoo) # 256*3
# Apos = torch.searchsorted(Acoo, Bcoo) # 128*3
# mask = B.flatten()[Apos] == A # 128*3

# Aint = mask.nonzero() # NNZ (flatten (i,j) of A)
# Bint = Apos[mask] # NNZ (flatten (j,k) of B)

# i = (torch.arange(128*3) // 3)[Aint] # NNZ -> extract i from Aint
# j = (torch.arange(256*3) // 3)[Bint] # NNZ : Bint.j
# ij = torch.stack([i,j], dim=1) # Aint.i, Bint.j

# unq_ij, counts = torch.unique(ij)
# cmask = counts == 3
# (unq_ij[cmask]
# test_case = {
#     "query": "A[i, c] == B[j, c] -> (i, j)",
#     "tensor_dims": {"A": (10, 5), "B": (10, 5)},
# }


# currently : table -> torch.all (o)
# test_case = {
#     "query": "A[i] == B[j, k] -> ()",
#     "tensor_dims": {"A": (10,), "B": (10, 10)},
# }

# currently : table (o) optimized (o)
# should reduce to intersection over [data, j] (reduce)
# test_case = {
#     "query": "A[i, j] == B[j, k] -> (i, j, k)",
#     "tensor_dims": {"A": (10, 10), "B": (10, 10)},
# }

# test_case = {
#     "query": "A[i] == B[j], B[i] == C[j]",
#     "tensor_dims": {"A": (10,), "B": (10,), "C": (10,)},
# }

# test_case = {
#     "query": "LA[i] <= RB[j], RA[i] <= LB[j]",
#     "tensor_dims": {"LA": (10,), "RB": (10,)},
# }

tensors = {
    name: torch.randint(size=dims, high=10, device=device)
    for name, dims in test_case["tensor_dims"].items()
}

with torch._inductor.utils.fresh_inductor_cache():
    compiled_intersect = torch.compile(table_intersect, dynamic=True)
    # compiled_intersect = table_intersect
    result = compiled_intersect(test_case["query"], **tensors, device=device)
