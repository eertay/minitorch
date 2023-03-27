# from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    current = vals[arg]

    vals_forward = []
    vals_backward = []

    for i in range(len(vals)):
        if i == arg:
            vals_forward.append(current + epsilon)
            vals_backward.append(current - epsilon)
        else:
            vals_forward.append(vals[i])
            vals_backward.append(vals[i])

    return (f(*vals_forward) - f(*vals_backward)) / (2 * epsilon)

    # return (inner_calc(arg + epsilon) - inner_calc(arg - epsilon)) / (2 * epsilon)
    # return (f(vals[arg] + epsilon) - f(vals[arg] - epsilon)) / (2 * epsilon)
    # return f(vals[arg] + (epsilon / 2)) - f(vals[arg] - (epsilon / 2))


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological ƒorder starting from the right.
    """

    visited_nodes = set()
    sorted: List[Variable] = []

    def dfs(variable: Variable) -> None:

        if variable.unique_id in visited_nodes:
            return

        if not variable.is_leaf():
            for neighbour in variable.parents:
                if not neighbour.is_constant():
                    dfs(neighbour)

        visited_nodes.add(variable.unique_id)
        sorted.insert(0, variable)

    dfs(variable)
    return sorted


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d

    # graph = topological_sort(variable)
    # dictionary = defaultdict(float)
    # dictionary[variable.unique_id] = deriv

    # for node in graph:

    #     current_deriv = dictionary[node.unique_id]

    #     if node.is_leaf():
    #         node.accumulate_derivative(current_deriv)

    #     else:
    #         for (value, derivative) in node.chain_rule(current_deriv):
    #             dictionary[value.unique_id] += derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
