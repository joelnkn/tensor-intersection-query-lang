from typing import List, Optional
from tiql.parsing.lexer import Token, TokenType
from tiql.matching.ast import (
    ASTNode,
    Query,
    QueryExpr,
    Access,
    FuncCall,
    BinaryOp,
    Constant,
    Identifier,
)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current_pos = 0

    def peek(self) -> Optional[Token]:
        if self.current_pos < len(self.tokens):
            return self.tokens[self.current_pos]
        return Token(TokenType.eof, "")  # End of tokens

    def expect(self, expected_type: TokenType, value: str = None) -> Token:
        token = self.peek()
        if token is None or token.token_type != expected_type:
            raise ValueError(f"Expected {expected_type, value}, but got {token}")

        if value is not None and token.value != value:
            raise ValueError(f"Expected {value}, but got {token.value}")

        return token

    def consume(
        self, expected_type: TokenType = None, value: str = None
    ) -> Optional[Token]:
        """
        Move to the next token and return the current one.
        """
        if expected_type is not None:
            self.expect(expected_type, value)
        token = self.peek()
        if token:
            self.current_pos += 1
        return token

    def parse(self) -> Query:
        return self.parse_query()

    def parse_query(self) -> Query:
        """
        query ::= q_expr | query, q_expr
        Handles both single q_expr and comma-separated lists.
        """
        expressions = [self.parse_q_expr()]

        while (
            self.peek()
            and self.peek().token_type == TokenType.punctuation
            and self.peek().value == ","
        ):
            self.consume()  # Consume comma
            expressions.append(self.parse_q_expr())

        return Query(expressions=expressions)

    def parse_q_expr(self) -> QueryExpr:
        """
        q_expr ::= expr query_op expr
        """
        left = self.parse_expr()
        op_token = self.consume(TokenType.query_op)  # relational op
        right = self.parse_expr()

        return QueryExpr(left=left, op=op_token.value, right=right)

    def parse_expr(self) -> ASTNode:
        """
        expr ::= term ([+-] term)*
        """
        # Parse +- with left associativity
        node = self.parse_term()
        while self.peek().token_type == TokenType.bin_op and (
            self.peek().value == "+" or self.peek().value == "-"
        ):
            op = self.consume()
            node = BinaryOp(left=node, op=op.value, right=self.parse_term())
        return node

    def parse_term(self) -> ASTNode:
        """
        expr ::= unit (* unit)*
        """
        # Parse * with left associativity
        node = self.parse_unit()
        while self.peek().token_type == TokenType.bin_op and self.peek().value == "*":
            op = self.consume()
            node = BinaryOp(left=node, op=op.value, right=self.parse_unit())
        return node

    def parse_unit(self) -> ASTNode:
        """
        expr ::= constant
               | func(expr, expr)
               | access
        """
        token = self.peek()

        if token.token_type == TokenType.func:
            return self.parse_func_call()

        if token.token_type == TokenType.identifier:
            return self.parse_access()

        if token.token_type == TokenType.constant:
            self.consume()
            return Constant(value=int(token.value))

        raise ValueError(f"Unexpected token when parsing expression: {token}")

    def parse_func_call(self) -> FuncCall:
        """
        func(expr, expr)
        """
        func_token = self.consume(TokenType.func)
        self.consume(TokenType.punctuation, "(")
        arg1 = self.parse_expr()
        self.consume(TokenType.punctuation, ",")
        arg2 = self.parse_expr()
        self.consume(TokenType.punctuation, ")")

        return FuncCall(func=func_token.value, args=[arg1, arg2])

    def parse_access(self) -> Access:
        """
        access ::= identifier[{position},+]
        """
        tensor = self.consume(TokenType.identifier)
        self.consume(TokenType.punctuation, "[")

        indices = [self.parse_position()]

        while (
            self.peek()
            and self.peek().token_type == TokenType.punctuation
            and self.peek().value == ","
        ):
            self.consume()  # Consume comma
            indices.append(self.parse_position())

        self.consume(TokenType.punctuation, "]")

        return Access(tensor=tensor.value, indices=indices)

    def parse_position(self) -> ASTNode:
        """
        position ::= access | identifier (simple index)
        """
        token = self.peek()

        if token.token_type == TokenType.identifier:
            if self._lookahead_is_punctuation("["):
                return self.parse_access()  # It's nested access
            else:
                self.consume()
                return token.value  # Simple identifier

        raise ValueError(f"Unexpected token: {token}")

    def _lookahead_is_punctuation(self, value: str) -> bool:
        """
        Look ahead one token and check if it's a punctuation with the given value.
        """
        next_token = self._lookahead()
        return (
            next_token
            and next_token.token_type == TokenType.punctuation
            and next_token.value == value
        )

    def _lookahead(self) -> Token:
        """
        Get the next token without consuming it.
        """
        if self.current_pos + 1 < len(self.tokens):
            return self.tokens[self.current_pos + 1]
        return Token(TokenType.eof, "")
