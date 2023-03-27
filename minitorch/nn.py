from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    t = (
        input.contiguous()
        .view(batch, channel, new_height, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )

    return t.view(batch, channel, new_height, new_width, kh * kw), new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    new_height = height * (width // kw)

    return input.contiguous().view(batch, channel, new_height, kw).sum(3).view(
        batch, channel, height, width // kw
    ).permute(0, 1, 3, 2).contiguous().view(
        batch, channel, height // kh * width // kw, kh
    ).sum(
        3
    ).view(
        batch, channel, height // kh, width // kw
    ) / (
        kh * kw
    )


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        noise = rand(input.shape) * 1e-6
        inp = input + noise
        dim_int = int(dim.item())
        ctx.save_for_backward(inp, dim_int)
        print(inp.shape)
        return max_reduce(inp, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    return input.exp() / input.exp().sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    return input - input.exp().sum(dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    return max_reduce(
        max_reduce(
            input.contiguous().view(batch, channel, height * (width // kw), kw), 3
        )
        .view(batch, channel, height, width // kw)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, channel, height // kh * width // kw, kh),
        3,
    ).view(batch, channel, height // kh, width // kw)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    if ignore:
        return input
    else:
        # Multiply input by a random tensor where the values are greater than the dropout rate
        return input * (rand(input.shape, backend=input.backend) > rate)
