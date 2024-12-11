from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_h = height // kh
    new_w = width // kw

    # Reshape and reorder the tensor
    tiled = input.contiguous().view(batch, channel, height, new_w, kw)
    tiled = tiled.permute(0, 1, 3, 2, 4)
    tiled = tiled.contiguous()
    tiled = tiled.view(batch, channel, new_h, new_w, kh * kw)

    return tiled, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, height, width = input.shape

    tiled, new_h, new_w = tile(input, kernel)

    pooled = tiled.mean(dim=4)

    return pooled.view(batch, channel, new_h, new_w)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max function which calculates the max value
        of the input tensor along the specified dimension.
        """
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Backward pass for max function which propagates the gradient
        to the input tensor along the specified dimension.
        """
        input, dim = ctx.saved_values
        return (argmax(input, int(dim.item())) * grad_output, dim)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    x = input.exp()
    return x / x.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    x_i = max(input, dim)
    return input - (input - x_i).exp().sum(dim=dim).log() - x_i


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, height, width = input.shape
    input, new_h, new_w = tile(input, kernel)
    out = max(input, 4)
    return out.view(batch, channel, new_h, new_w)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise"""
    if ignore:
        return input
    if rate >= 1.0:
        return input * 0
    if rate <= 0.0:
        return input
    mask = rand(input.shape) > rate
    return input * mask
