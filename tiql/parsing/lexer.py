import re
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass


class TokenType(Enum):
    punctuation = 1  # Brackets, commas, parens
    bin_op = 2  # Arithmetic operators: +, -, *
    query_op = 3  # Relational operators: <, <=, ==, >, >=
    func = 4  # min, max
    identifier = 5  # Tensors, indices like A, B, i, j
    constant = 6  # number literal value
    eof = 7  # represents the end of file


# Regex map for each token type
token_regex = {
    # Punctuation - brackets, parens, commas, right arrow
    TokenType.punctuation: r"([()\[\],]|->)",
    # Binary operators - arithmetic
    TokenType.bin_op: r"[+\-*]",
    # Query operators - relational
    TokenType.query_op: r"(<=|>=|==|<|>)",
    # Functions - min, max
    TokenType.func: r"\b(min|max)\b",
    # Identifiers - tensor names, indices
    TokenType.identifier: r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
    # integer constants, optional sign
    TokenType.constant: r"[+-]?\d+",
}


@dataclass
class Token:
    token_type: TokenType
    value: str


class Lexer:
    def __init__(self, query: str):
        self.query = query
        self.position = 0
        self.tokens = []

    def tokenize(self) -> List[Token]:
        while self.position < len(self.query):
            if self.query[self.position].isspace():
                self.position += 1
                continue  # skip all whitespace

            matched_token = self._match_token()

            if matched_token is None:
                raise ValueError(
                    f"Unexpected character at position {self.position}: {self.query[self.position]}"
                )

            self.tokens.append(matched_token)

        return self.tokens

    def _match_token(self) -> Optional[Token]:
        for token_type, pattern in token_regex.items():
            # regex = re.compile(pattern)
            # match = regex.match(self.query, self.position)

            match = re.match(pattern, self.query[self.position :])

            if match:
                value = match.group(0)
                self.position += len(value)  # Advance the position
                return Token(token_type, value)

        # No match found, this triggers the ValueError in tokenize()
        return None
