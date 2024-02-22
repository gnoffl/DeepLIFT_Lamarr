import torch
import src.deeplift.utils as utils
from typing import Tuple, List
from functools import reduce
import operator

from src.deeplift import constants


def get_attributes_avg_pool(layer: torch.jit._script.RecursiveScriptModule) -> \
        Tuple[int, List[int], List[int], List[int]]:
    """
    gets the attributes of the current avgpool layer

    Args:
        layer: the avgpool layer

    Returns:
        returns tuple (dimensionality, kernel_size, stride, padding) of the avgpool layer where dimensionality is
        the number of dimensions for the avgpool layer. kernel_size, stride and padding are lists of length
        dimensionality, each containing the kernel size, stride or padding for the corresponding dimension
    """
    dimensionality = int(layer.original_name.split("Pool")[-1][0])
    kernel_size_net = layer.kernel_size
    if not isinstance(kernel_size_net, tuple):
        kernel_size = [kernel_size_net for _ in range(dimensionality)]
    elif len(kernel_size_net) != dimensionality:
        raise ValueError("kernel size must be of length dimensionality")
    else:
        kernel_size = list(kernel_size_net)

    stride_net = layer.stride
    if not isinstance(stride_net, tuple):
        stride = [stride_net for _ in range(dimensionality)]
    elif len(stride_net) != dimensionality:
        raise ValueError("stride must be of length dimensionality")
    else:
        stride = list(stride_net)
    padding_net = layer.padding
    if not isinstance(padding_net, tuple):
        padding = [padding_net for _ in range(dimensionality)]
    elif len(padding_net) != dimensionality:
        raise ValueError("padding must be of length dimensionality")
    else:
        padding = list(padding_net)
    return dimensionality, kernel_size, padding, stride


def get_ranges(dimensionality: int, output_shape: torch.Size) -> List[range]:
    """
    gets the ranges needed for iterating over the results of an avgpooling layer

    Args:
        dimensionality: dimensionality of the avgpooling layer
        output_shape: outputs of the avgpooling layer

    Returns:
        List of ranges for iterating over the results of the avgpooling layer. Ranges walk over the shape of the
        multipliers of the previous layer. List always contains 3 elements, if a dimension does not exist, the
        corresponding element is a range over a single negative number
    """
    ranges = [range(output_shape[3])]
    if dimensionality > 1:
        ranges.append(range(output_shape[4]))
    else:
        ranges.append(range(-2, -1))
    if dimensionality > 2:
        ranges.append(range(output_shape[5]))
    else:
        ranges.append(range(-2, -1))
    return ranges


def get_kernel_pos_slices(i: int, j: int, k: int, kernel_size: List[int], stride: List[int]) -> Tuple[slice]:
    """
    gets the slice objects to select the elements of the input tensor to the AvgPoolLayer inside the kernel
    responsible for calculating the output at position i,j,k

    Args:
        i: first data position in the output
        j: second data position in the output or negative number indicating, that no further dimension exists
        k: third data position in the output or negative number indicating, that no further dimension exists
        kernel_size: size of the kernel
        stride: stride of the kernel

    Returns:
        tuple of slice objects to select the elements inside the kernel responsible for calculating the output
    """
    slices = [
        slice(None, None),
        slice(None, None),
        slice(None, None),
        slice(i * stride[0], i * stride[0] + kernel_size[0])
    ]
    if j > -1:
        slices.append(slice(j * stride[1], j * stride[1] + kernel_size[1]))
    if k > -1:
        slices.append(slice(k * stride[2], k * stride[2] + kernel_size[2]))
    return tuple(slices)


def get_avg_pos_slices(i: int, j: int, k: int) -> Tuple:
    """
    gets the slice objects to select the element at position i,j,k

    Args:
        i: first data position in the output
        j: second data position in the output or negative number indicating, that no further dimension exists
        k: third data position in the output or negative number indicating, that no further dimension exists

    Returns:
        tuple of the slice objects to select the element at position i,j,k
    """
    slices = [
        slice(None, None),
        slice(None, None),
        slice(None, None),
        i
    ]
    if j > -1:
        slices.append(j)
    if k > -1:
        slices.append(k)
    return tuple(slices)


def remove_padding(padded: torch.Tensor, padding: List[int]) -> torch.Tensor:
    """
    removes padding from a tensor

    Args:
        padded: the padded tensor
        padding: List of padding values

    Returns:
        the tensor without padding
    """
    slices = [slice(None, None), slice(None, None)]
    for i in range(len(padding)):
        if padding[i] != 0:
            slices.append(slice(padding[i], -padding[i]))
        else:
            slices.append(slice(None, None))
    unpadded = padded[tuple(slices)]
    return unpadded


def apply_padding(padding: List[int], to_pad: torch.Tensor) -> torch.Tensor:
    """
    applies padding to a tensor

    Args:
        padding: List of padding values
        to_pad: tensor to apply padding to

    Returns:
        padded tensor
    """
    padding_list = []
    for i in range(len(padding)):
        padding_list.append(padding[i])
        padding_list.append(padding[i])
    padded_activation = torch.nn.functional.pad(to_pad, tuple(padding_list))
    return padded_activation


def reverse_averaging(dimensionality: int, input_tensor: torch.Tensor, multipliers: torch.Tensor,
                      kernel_size: List[int], stride: List[int]) -> torch.Tensor:
    """
    calculates the reverse effect of an avgpooling layer and applies it to a tensor of the shape of the output of
    the avgpooling layer

    Args:
        dimensionality: dimensionality of the avgpooling layer
        input_tensor: tensor that was used as the input of the avgpooling layer
        multipliers: multipliers that are to be propagated backwards through the avgpooling layer
        kernel_size: size of the kernel
        stride: stride of the kernel

    Returns:
        the multipliers propagated backwards through the avgpooling layer
    """
    # create empty output tensor with the same shape as the output from the previous layer (except for the batch
    # dimension, because this is determined by the size of the output on the backward pass)
    output = torch.zeros_like(input_tensor)
    output = utils.repeat_to_match_shapes(for_first_dim=multipliers, to_repeat=output)
    ranges = get_ranges(dimensionality=dimensionality, output_shape=multipliers.shape)
    # loops walk over the previous multipliers, while the stride and kernel size are used to calculate the
    # corresponding in the previous layer activation
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                # weights for each position in the kernel are the same: 1 / (kernel_size_total)
                # the correct position for output and multipliers are selected by the slicing indices
                slicing_indices = get_kernel_pos_slices(i, j, k, kernel_size, stride)
                kernel_size_total = reduce(operator.mul, kernel_size, 1)
                avg_slice = get_avg_pos_slices(i, j, k)
                division_result = multipliers[avg_slice] / kernel_size_total
                division_result = torch.reshape(division_result, division_result.shape + (1, 1))
                output[slicing_indices] = division_result
    return output


def avgpool_pos_neg(current_layer: torch.jit._script.RecursiveScriptModule, prev_layer_diff: torch.Tensor,
                    previous_multipliers: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Propagates positive and negative multipliers backwards through a average pool layer.

    Args:
        current_layer: average pool layer to propagate multipliers through
        prev_layer_diff: difference from reference from the layer one step closer to the input
        previous_multipliers: tuple of multipliers (A, B) to be propagated through the layer. A are the positive
        multipliers and B are the negative multipliers.

    Returns:
        Single tensor containing the propagated multipliers.
    """
    pos_multipliers, neg_multipliers = previous_multipliers
    pos_neg_avg = (pos_multipliers + neg_multipliers) / 2
    # get attributes of current avgpool layer
    dimensionality, kernel_size, padding, stride = get_attributes_avg_pool(current_layer)
    # apply padding to the previous layer activation to simulate the padding that was applied in the forward pass
    prev_layer_diff = apply_padding(padding=padding, to_pad=prev_layer_diff)

    pos_diff_from_ref_mask = prev_layer_diff > constants.ZERO_MASK_THRESHOLD
    neg_diff_from_ref_mask = prev_layer_diff < -constants.ZERO_MASK_THRESHOLD
    zero_diff_from_ref_mask = torch.abs(prev_layer_diff) <= constants.ZERO_MASK_THRESHOLD

    # calculate the multipliers of the previous layer activations to the output of the current layer
    # combinations of values are essentially the same as for the convolutional layer. The only difference is that
    # the weights are always positive for avgpool.
    pos_pos = reverse_averaging(dimensionality=dimensionality, kernel_size=kernel_size, stride=stride,
                                input_tensor=prev_layer_diff, multipliers=pos_multipliers)
    pos_pos = remove_padding(padded=pos_pos, padding=padding)
    pos_pos = pos_pos * pos_diff_from_ref_mask

    pos_neg = reverse_averaging(dimensionality=dimensionality, input_tensor=prev_layer_diff, stride=stride,
                                kernel_size=kernel_size, multipliers=neg_multipliers)
    pos_neg = remove_padding(padded=pos_neg, padding=padding)
    pos_neg = pos_neg * neg_diff_from_ref_mask

    zeros = reverse_averaging(dimensionality=dimensionality, input_tensor=prev_layer_diff, stride=stride,
                              multipliers=pos_neg_avg, kernel_size=kernel_size)
    zeros = remove_padding(padded=zeros, padding=padding)
    zeros = zeros * zero_diff_from_ref_mask

    output = pos_pos + pos_neg + zeros
    return output


def avgpool(dl, current_layer: torch.jit._script.RecursiveScriptModule, prev_layer_diff: torch.Tensor,
            previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    calculates the multipliers for the current avgpool layer by calculating the reverse effects of the avgpooling

    Args:
        dl: DeepLiftClass object containing the model and the forward and reference activations
        current_layer: current avgpool layer to calculate the multipliers for
        prev_layer_diff: outputs from the forward pass of the input for the layer one closer to the input
        previous_multipliers: multipliers of the previous layer (one closer to the output)

    Returns:
        multipliers for the current layer
    """
    if not hasattr(dl, "reference_activation"):
        raise AttributeError("reference activation was not set yet")

    if not isinstance(previous_multipliers, torch.Tensor) and not isinstance(previous_multipliers, tuple):
        raise ValueError("previous multipliers must be single tensor (if rescale is used) or tuple of"
                         "tensors (if reveal_cancel was used)!\n"
                         "Received type was {}".format(type(previous_multipliers)))

    #case if reveal cancel was used
    if isinstance(previous_multipliers, tuple):
        current_multipliers = avgpool_pos_neg(current_layer=current_layer,
                                              prev_layer_diff=prev_layer_diff,
                                              previous_multipliers=previous_multipliers)
        return current_multipliers

    # get attributes of current avgpool layer
    dimensionality, kernel_size, padding, stride = get_attributes_avg_pool(current_layer)
    # apply padding to the previous layer activation to simulate the padding that was applied in the forward pass
    prev_layer_diff = apply_padding(padding=padding, to_pad=prev_layer_diff)
    # calculate the multipliers of the previous layer activations to the output of the current layer
    output = reverse_averaging(dimensionality=dimensionality, kernel_size=kernel_size, stride=stride,
                               input_tensor=prev_layer_diff, multipliers=previous_multipliers)
    # remove padding from the output
    output = remove_padding(padded=output, padding=padding)
    return output
