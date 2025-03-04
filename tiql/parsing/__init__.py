from .lexer import Lexer, TokenType
from .parser import Parser
from .ast import Query, QueryExpr, Access, Identifier, Constant, BinaryOp, FuncCall

__all__ = [
    Lexer,
    Parser,
    TokenType,
    Query,
    QueryExpr,
    Access,
    Identifier,
    Constant,
    BinaryOp,
    FuncCall,
]


def parse(query: str) -> Query:
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
