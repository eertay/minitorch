"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiply
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        Product.
    """
    return x * y


def id(x: float) -> float:
    """
    Identity
    Args:
        x (float): numeric value
    Returns:
        The value itself.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Addition
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        Sum.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negative
    Args:
        x (float): numeric value
    Returns:
        Opposite.
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Less than operation
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        1.0 if x is less than y else 0.0.
    """
    if x < y:
        return 1.0

    else:
        return 0.0


def eq(x: float, y: float) -> float:
    """
    Is equal to operation
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        1.0 if x is equal to y else 0.0.
    """
    if x == y:
        return 1.0

    else:
        return 0.0


def max(x: float, y: float) -> float:
    """
    Maximum
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        Maximum of the two values.
    """

    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    """
    Is close
    Args:
        x (float): numeric value
        y (float): numeric value
    Returns:
        1.0 if difference less than 1e-2 else 0.0.
    """

    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    r"""
    Sigmoid
    Args:
        x (float): numeric value
    Returns:
        Sigmoid value.
    """
    if x >= 0:
        return float(1.0 / (1.0 + math.exp(-x)))

    else:
        return float(math.exp(x) / (1.0 + math.exp(x)))


def relu(x: float) -> float:
    """
    Relu
    Args:
        x (float): numeric value
    Returns:
        Relu output.
    """
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    """
    Args:
        x (float): numeric value
        d (float): numeric value
    Returns:
        Log derivative times d.
    """

    if x == 0:
        return 0.0
    else:
        return (1 / x) * d


def inv(x: float) -> float:
    """
    Inverse
    Args:
        x (float): numeric value
    Returns:
        Inverse.
    """
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    """
    Args:
        x (float): numeric value
        d (float): numeric value
    Returns:
        Inverse derivative times d.
    """

    return ((-1 / x**2)) * d


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    """
    Args:
        x (float): numeric value
        d (float): numeric value
    Returns:
        Relu derivative times d.
    """

    if x <= 0:
        return 0.0
    else:
        return d


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def my_map(iter_list: Iterable[float]) -> Iterable[float]:
        return [fn(i) for i in iter_list]

    return my_map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negative values of all elements in a list
    Args:
        ls: a list
    Returns:
        A list containing all negative values of ls
    """
    my_func = map(neg)
    return my_func(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def my_zip(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        zipped = zip(ls1, ls2)
        return [fn(elem1, elem2) for elem1, elem2 in zipped]

    return my_zip


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Adds two lists (concat)
    Args:
        ls1: a list
        ls2: a list
    Returns:
        A new list that combines elements from both ls1 and ls2
    """
    my_func = zipWith(add)
    return my_func(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def my_reduce(ls: Iterable[float]) -> float:
        curr = start
        for element in ls:
            curr = fn(element, curr)

        return curr

    return my_reduce


def sum(ls: Iterable[float]) -> float:
    """
    Sum all the values in a list
    Args:
        ls: a list
    Returns:
        A float of the sum of all elements in ls
    """
    # if len(ls) == 0:
    #     return 0

    my_sum = reduce(add, 0)
    return my_sum(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Product of all values in a list
    Args:
        ls: a list
    Returns:
        A float of the product of all elements in ls
    """

    my_prod = reduce(mul, 1)
    return my_prod(ls)
