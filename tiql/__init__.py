from .solver import Solver, intersect, table_intersect, simple_intersect
import logging

logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

__all__ = ["Solver", "intersect", "table_intersect", "simple_intersect"]
