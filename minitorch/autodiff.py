from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)
    vals_list[arg] += epsilon
    f_plus = f(*vals_list)
    vals_list[arg] -= 2 * epsilon
    f_minus = f(*vals_list)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Check if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the partial derivatives of this variable with respect to its parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited: set[Variable] = set()
    order: list[Variable] = []

    def visit(var: Variable) -> None:
        if var not in visited and not var.is_constant():
            visited.add(var)
            for parent in var.parents:
                visit(parent)
            order.append(var)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable (Variable): The variable to backpropagate from.
        deriv (Any): The derivative to propagate.

    """
    derivatives = {variable: deriv}
    for var in list(topological_sort(variable)):
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var])
        else:
            for parent, grad in var.chain_rule(derivatives[var]):
                if parent not in derivatives:
                    derivatives[parent] = grad
                else:
                    derivatives[parent] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors from the forward pass."""
        return self.saved_values
