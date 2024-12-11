from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch
import math
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the input valuesvals"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    r"""Logarithm function $f(x) = \log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    r"""Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    r"""Inverse function $f(x) = \frac{1}{x}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation."""
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation."""
        return float(operators.neg(d_output))


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid."""
        (a,) = ctx.saved_values
        return float(operators.sigmoid_back(a, d_output))


class ReLU(ScalarFunction):
    """ReLU function $f(x) =$ max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU."""
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU."""
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential."""
        result = math.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential."""
        (result,) = ctx.saved_values
        return d_output * result


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than."""
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equal."""
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equal."""
        return 0.0, 0.0
