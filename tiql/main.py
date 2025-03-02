import sys
from tiql.lexer import Lexer
from tiql.parser import Parser
from tiql.semantic import SemanticAnalyzer
from tiql.codegen import CodeGenerator
from tiql.executor import QueryExecutor


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.main <input_query_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, "r") as f:
        query = f.read()

    # Pipeline: Lexing -> Parsing -> Semantic Analysis -> Code Generation -> Execution
    lexer = Lexer()
    tokens = lexer.tokenize(query)

    parser = Parser()
    ast = parser.parse(tokens)

    analyzer = SemanticAnalyzer()
    analyzer.analyze(ast)

    codegen = CodeGenerator()
    code = codegen.generate(ast)

    executor = QueryExecutor()
    result = executor.execute(code)

    print("Query Result:")
    print(result)


if __name__ == "__main__":
    main()
