from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np
from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __hash__(self):
        return hash(self.unique_id)

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable was created by an operation on constants."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Iterable of the parent variables of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the derivatives of the parent variables.

        Args:
        ----
            d_output: The derivative of the current variable with respect to the output.

        Returns:
        -------
            An iterable of tuples, where each tuple contains a parent variable and its derivative.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn._backward(h.ctx, d_output)
        return zip(h.inputs, grads)

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Adds two Scalar objects or a Scalar and a number.

        Args:
        ----
            b: The other Scalar or number to add.

        Returns:
        -------
            A new Scalar representing the sum.

        """
        return Add.apply(self, b)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compares if this Scalar is less than another Scalar or number.

        Args:
        ----
            b: The other Scalar or number to compare.

        Returns:
        -------
            A new Scalar representing the comparison result.

        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Compares if this Scalar is greater than another Scalar or number.

        Args:
        ----
            b: The other Scalar or number to compare.

        Returns:
        -------
            A new Scalar representing the comparison result.

        """
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        """Compares if this Scalar is equal to another Scalar or number.

        Args:
        ----
            b: The other Scalar or number to compare.

        Returns:
        -------
            A new Scalar representing the comparison result.

        """
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtracts another Scalar or number from this Scalar.

        Args:
        ----
            b: The other Scalar or number to subtract.

        Returns:
        -------
            A new Scalar representing the difference.

        """
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negates this Scalar.

        Returns
        -------
            A new Scalar representing the negation.

        """
        return Neg.apply(self)

    def log(self) -> Scalar:
        """Computes the natural logarithm of this Scalar.

        Returns
        -------
            A new Scalar representing the logarithm.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Computes the exponential of this Scalar.

        Returns
        -------
            A new Scalar representing the exponential.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Computes the sigmoid of this Scalar.

        Returns
        -------
            A new Scalar representing the sigmoid.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Computes the ReLU (Rectified Linear Unit) of this Scalar.

        Returns
        -------
            A new Scalar representing the ReLU.

        """
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
