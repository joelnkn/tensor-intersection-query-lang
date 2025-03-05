from .lexer import Lexer, TokenType
from .parser import Parser
from tiql.matching.ast import Query


def parse(query: str) -> Query:
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
