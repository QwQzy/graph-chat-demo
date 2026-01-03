"""
math_tools.py

A small, LLM-friendly math tool collection for LangChain / LangGraph.
All tools include clear schemas (docstrings) designed for tool calling.
"""

from typing import Union, List, Dict
import math
from langchain.tools import tool, BaseTool

Number = Union[int, float]


@tool
def add(a: Number, b: Number) -> Number:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def subtract(a: Number, b: Number) -> Number:
    """Subtract `b` from `a`.

    Args:
        a: The minuend.
        b: The subtrahend.
    """
    return a - b


@tool
def multiply(a: Number, b: Number) -> Number:
    """Multiply `a` and `b`.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


@tool
def divide(a: Number, b: Number) -> float:
    """Divide `a` by `b`.

    Args:
        a: The dividend.
        b: The divisor. Must not be zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@tool
def power(base: Number, exponent: Number) -> Number:
    """Raise `base` to the power of `exponent`.

    Args:
        base: The base number.
        exponent: The exponent.
    """
    return base ** exponent


@tool
def sqrt(x: Number) -> float:
    """Calculate the square root of `x`.

    Args:
        x: A non-negative number.
    """
    if x < 0:
        raise ValueError("Square root of a negative number is not allowed.")
    return math.sqrt(x)


@tool
def abs_val(x: Number) -> Number:
    """Return the absolute value of `x`.

    Args:
        x: A number.
    """
    return abs(x)


@tool
def max_val(a: Number, b: Number) -> Number:
    """Return the larger of two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return max(a, b)


@tool
def min_val(a: Number, b: Number) -> Number:
    """Return the smaller of two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return min(a, b)


@tool
def average(values: List[Number]) -> float:
    """Calculate the average of a list of numbers.

    Args:
        values: A non-empty list of numbers.
    """
    if not values:
        raise ValueError("Values list cannot be empty.")
    return sum(values) / len(values)


# Tool registry (useful for manual binding or inspection)
tool_map: Dict[str, BaseTool] = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "power": power,
    "sqrt": sqrt,
    "abs_val": abs_val,
    "max_val": max_val,
    "min_val": min_val,
    "average": average,
}
