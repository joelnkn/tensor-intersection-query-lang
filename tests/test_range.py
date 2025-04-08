# import torch
# from tiql.matching.range import Range


# # region index intersect
# def test_index_intersect_1d():
#     indices_a = torch.tensor([[0, 1, 2, 3]])
#     indices_b = torch.tensor([[2, 3, 4, 5]])
#     symbols_a = {"i": 0}
#     symbols_b = {"i": 0}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.tensor([[2, 0], [3, 1]])

#     assert torch.equal(result, expected)


# def test_index_intersect_basic():
#     indices_a = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
#     indices_b = torch.tensor([[2, 3, 4, 5], [12, 13, 14, 15]])
#     symbols_a = {"i": 0, "j": 1}
#     symbols_b = {"i": 0, "j": 1}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.tensor([[2, 0], [3, 1]])  # Assuming exact index matching

#     assert torch.equal(result, expected)


# def test_index_intersect_no_match():
#     indices_a = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
#     indices_b = torch.tensor([[4, 5, 6, 7], [14, 15, 16, 17]])
#     symbols_a = {"i": 0, "j": 1}
#     symbols_b = {"i": 0, "j": 1}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.empty((0, 2), dtype=torch.int64)  # No matches

#     assert torch.equal(result, expected)


# def test_index_intersect_empty():
#     indices_a = torch.empty((2, 0), dtype=torch.int64)
#     indices_b = torch.empty((2, 0), dtype=torch.int64)
#     symbols_a = {"i": 0, "j": 1}
#     symbols_b = {"i": 0, "j": 1}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.empty((0, 2), dtype=torch.int64)

#     assert torch.equal(result, expected)


# def test_index_intersect_with_duplicates():
#     indices_a = torch.tensor([[1, 2, 2, 3, 3, 3], [10, 20, 20, 30, 30, 30]])
#     indices_b = torch.tensor([[2, 3, 3, 4, 5, 3], [20, 30, 30, 40, 50, 30]])
#     symbols_a = {"i": 0, "j": 1}
#     symbols_b = {"i": 0, "j": 1}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.tensor(
#         [[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1], [5, 2]]
#     )

#     assert torch.equal(result, expected)


# def test_index_intersect_with_offset_symbols():
#     indices_a = torch.tensor([[1, 2, 2, 3, 3, 3], [10, 20, 20, 30, 30, 60]])
#     indices_b = torch.tensor([[20, 30, 30, 40, 50, 30], [2, 3, 3, 4, 5, 3]])
#     symbols_a = {"i": 0, "j": 1}
#     symbols_b = {"j": 0, "k": 1}
#     device = torch.device("cpu")

#     range_a = Range(indices_a, symbols_a, device)
#     range_b = Range(indices_b, symbols_b, device)

#     result = range_a.index_intersect(range_b)
#     expected = torch.tensor(
#         [[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [3, 5], [4, 1], [4, 2], [4, 5]]
#     )

#     assert torch.equal(result, expected)


# # endregion
