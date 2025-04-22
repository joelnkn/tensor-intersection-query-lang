from tiql.parsing import Lexer, Parser
from tiql.matching.ast import (
    Query,
    QueryExpr,
    Access,
    Constant,
    BinaryOp,
    FuncCall,
)


def test_simple_equality():
    query = "A[i] == B[j]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access(tensor="A", indices=["i"]),
                op="==",
                right=Access(tensor="B", indices=["j"]),
            )
        ],
        out_indices=("i", "j"),
    )

    assert ast == expected


def test_chained_query():
    query = "A[i] == B[j], C[k] >= D[l]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access(tensor="A", indices=["i"]),
                op="==",
                right=Access(tensor="B", indices=["j"]),
            ),
            QueryExpr(
                left=Access(tensor="C", indices=["k"]),
                op=">=",
                right=Access(tensor="D", indices=["l"]),
            ),
        ],
        out_indices=("i", "j", "k", "l"),
    )

    assert ast == expected


def test_binary_expression():
    query = "A[i] + B[j] >= C[i]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=BinaryOp(
                    left=Access("A", ["i"]),
                    op="+",
                    right=Access("B", ["j"]),
                ),
                op=">=",
                right=Access("C", ["i"]),
            )
        ],
        out_indices=("i", "j"),
    )

    assert ast == expected


def test_function_call():
    query = "min(A[i], B[j]) >= max(C[i], D[j])"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=FuncCall(
                    func="min",
                    args=[
                        Access("A", ["i"]),
                        Access("B", ["j"]),
                    ],
                ),
                op=">=",
                right=FuncCall(
                    func="max",
                    args=[
                        Access("C", ["i"]),
                        Access("D", ["j"]),
                    ],
                ),
            )
        ],
        out_indices=("i", "j"),
    )

    assert ast == expected


def test_nested_access():
    query = "A[i, B[j]] == C[k]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access("A", ["i", Access("B", ["j"])]),
                op="==",
                right=Access("C", ["k"]),
            )
        ],
        out_indices=("i", "j", "k"),
    )

    assert ast == expected


def test_constant():
    query = "5 * A[i] < B[j] + 2"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=BinaryOp(left=Constant(5), op="*", right=Access("A", ["i"])),
                op="<",
                right=BinaryOp(left=Access("B", ["j"]), op="+", right=Constant(2)),
            )
        ],
        out_indices=("i", "j"),
    )

    assert ast == expected


def parse_query(query: str) -> Query:
    """Helper to run full lexer + parser pipeline for tests"""
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
