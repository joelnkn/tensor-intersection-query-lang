from tiql.parsing import Lexer, TokenType


def test_lexer():
    query = "A[i] == B[j], min(A[i], B[j]) >= max(C[k], D[m])"
    lexer = Lexer(query)
    tokens = lexer.tokenize()

    expected_tokens = [
        (TokenType.identifier, "A"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "i"),
        (TokenType.punctuation, "]"),
        (TokenType.query_op, "=="),
        (TokenType.identifier, "B"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "j"),
        (TokenType.punctuation, "]"),
        (TokenType.punctuation, ","),
        (TokenType.func, "min"),
        (TokenType.punctuation, "("),
        (TokenType.identifier, "A"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "i"),
        (TokenType.punctuation, "]"),
        (TokenType.punctuation, ","),
        (TokenType.identifier, "B"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "j"),
        (TokenType.punctuation, "]"),
        (TokenType.punctuation, ")"),
        (TokenType.query_op, ">="),
        (TokenType.func, "max"),
        (TokenType.punctuation, "("),
        (TokenType.identifier, "C"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "k"),
        (TokenType.punctuation, "]"),
        (TokenType.punctuation, ","),
        (TokenType.identifier, "D"),
        (TokenType.punctuation, "["),
        (TokenType.identifier, "m"),
        (TokenType.punctuation, "]"),
        (TokenType.punctuation, ")"),
    ]

    assert [(t.token_type, t.value) for t in tokens] == expected_tokens
