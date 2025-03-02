from tiql.lexer import Lexer
from tiql.parser import Parser
from tiql.ast import Query, QueryExpr, Access, Identifier, Constant, BinaryOp, FuncCall


def test_simple_equality():
    query = "A[i] == B[j]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access(tensor="A", indices=[Identifier("i")]),
                op="==",
                right=Access(tensor="B", indices=[Identifier("j")]),
            )
        ]
    )

    assert ast == expected


def test_chained_query():
    query = "A[i] == B[j], C[k] >= D[l]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access(tensor="A", indices=[Identifier("i")]),
                op="==",
                right=Access(tensor="B", indices=[Identifier("j")]),
            ),
            QueryExpr(
                left=Access(tensor="C", indices=[Identifier("k")]),
                op=">=",
                right=Access(tensor="D", indices=[Identifier("l")]),
            ),
        ]
    )

    assert ast == expected


def test_binary_expression():
    query = "A[i] + B[j] >= C[i]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=BinaryOp(
                    left=Access("A", [Identifier("i")]),
                    op="+",
                    right=Access("B", [Identifier("j")]),
                ),
                op=">=",
                right=Access("C", [Identifier("i")]),
            )
        ]
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
                        Access("A", [Identifier("i")]),
                        Access("B", [Identifier("j")]),
                    ],
                ),
                op=">=",
                right=FuncCall(
                    func="max",
                    args=[
                        Access("C", [Identifier("i")]),
                        Access("D", [Identifier("j")]),
                    ],
                ),
            )
        ]
    )

    assert ast == expected


def test_nested_access():
    query = "A[i, B[j]] == C[k]"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=Access("A", [Identifier("i"), Access("B", [Identifier("j")])]),
                op="==",
                right=Access("C", [Identifier("k")]),
            )
        ]
    )

    assert ast == expected


def test_constant():
    query = "5 * A[i] < B[j] + 2"
    ast = parse_query(query)

    expected = Query(
        [
            QueryExpr(
                left=BinaryOp(
                    left=Constant(5), op="*", right=Access("A", [Identifier("i")])
                ),
                op="<",
                right=BinaryOp(
                    left=Access("B", [Identifier("j")]), op="+", right=Constant(2)
                ),
            )
        ]
    )

    assert ast == expected


def parse_query(query: str) -> Query:
    """Helper to run full lexer + parser pipeline for tests"""
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
