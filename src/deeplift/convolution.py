import warnings
import torch
import torch.nn as nn
from typing import Tuple

import deeplift.utils as utils


def set_up_conv_transpose_layer(current_layer: torch.jit._script.RecursiveScriptModule,
                                layer_weights: nn.Parameter) \
        -> nn.ConvTranspose1d or nn.ConvTranspose2d or nn.ConvTranspose3d:
    """
    creates a fitting convtranspose layer for the current conv layer

    Args:
        current_layer: the conv layer for which the reverse should be calculated
        layer_weights: weights of the current conv layer

    Returns:
        the initialized convtranspose layer
    """
    # get attributes of current conv layer
    stride = current_layer.stride
    padding = current_layer.padding
    padding_mode = current_layer.padding_mode
    if padding_mode != "zeros":
        warnings.warn("padding mode is not zero, this might lead to wrong results")
    dilation = current_layer.dilation
    groups = current_layer.groups
    kernel_size = current_layer.kernel_size
    dimensions = int(current_layer.original_name.split("Conv")[1][0])
    # determine reverse function
    if dimensions == 1:
        reverse_conv_layer = nn.ConvTranspose1d
    elif dimensions == 2:
        reverse_conv_layer = nn.ConvTranspose2d
    elif dimensions == 3:
        reverse_conv_layer = nn.ConvTranspose3d
    else:
        raise ValueError("dimensions must be 1,2 or 3")
    # create convtranspose layer with same attributes
    conv_transpose = reverse_conv_layer(in_channels=current_layer.out_channels,
                                        out_channels=current_layer.in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups,
                                        bias=False)

    # set weights of convtranspose layer to weights of current conv layer
    conv_transpose.weight = nn.Parameter(layer_weights)
    return conv_transpose


def linear_rule_conv_pos_neg(dl, previous_multipliers: Tuple[torch.Tensor, torch.Tensor],
                             current_layer: torch.jit._script.RecursiveScriptModule, current_layer_name: str,
                             reverse_conv: torch.nn.Module) -> torch.Tensor:
    """
    calculates the multipliers for the current conv layer using the linear rule for positive and negative
    multipliers.

    Args:
        dl: DeepLiftClass object containing the model and the reference activations
        previous_multipliers: multipliers of the previous layer to be propagated in shape (A, B), where A are the
            positive and B the negative multipliers
        current_layer: layer object for which the multipliers should be calculated
        current_layer_name: name of the layer for which the multipliers should be calculated
        reverse_conv: transposed convolution layer that reverts the effects of the current conv layer

    Returns:
        tensor with the multipliers propagated through the current layer
    """
    diff_from_ref_in = dl.diff_from_ref[dl.model.get_previous_layer_name(current_layer_name)]
    weights = current_layer.weight
    pos_multipliers, neg_multipliers = previous_multipliers
    neg_diff_from_ref_mask, pos_diff_from_ref_mask, neutral_diff_from_ref_mask, pos_weight_mask, neg_weight_mask = utils.get_pos_neg_masks(
        weights=weights, diff_from_ref_in=diff_from_ref_in, for_shape=pos_multipliers)

    # positive multiplier is used for positive combination of weights and differences (++, --)
    # negative multiplier is used for negative combination of weights and differences (-+, +-)
    # pos weights:
    pos_weights = weights * pos_weight_mask
    reverse_conv.weight = nn.Parameter(pos_weights)
    m_pos_pos = additional_dim_reverse_convolution(conv_transpose=reverse_conv,
                                                   previous_multipliers=pos_multipliers)
    m_pos_pos = m_pos_pos * pos_diff_from_ref_mask
    m_neg_pos = additional_dim_reverse_convolution(conv_transpose=reverse_conv,
                                                   previous_multipliers=neg_multipliers)
    m_neg_pos = m_neg_pos * neg_diff_from_ref_mask

    # negative weights:
    reverse_conv.weight = nn.Parameter(weights * neg_weight_mask)
    m_pos_neg = additional_dim_reverse_convolution(conv_transpose=reverse_conv,
                                                   previous_multipliers=neg_multipliers)
    m_pos_neg = m_pos_neg * pos_diff_from_ref_mask
    m_neg_neg = additional_dim_reverse_convolution(conv_transpose=reverse_conv,
                                                   previous_multipliers=pos_multipliers)
    m_neg_neg = m_neg_neg * neg_diff_from_ref_mask

    # special case of zero difference as defined in the paper with normal weights
    reverse_conv.weight = nn.Parameter(weights)
    m_neutral = additional_dim_reverse_convolution(conv_transpose=reverse_conv,
                                                   previous_multipliers=(pos_multipliers + neg_multipliers) / 2)
    m_neutral = m_neutral * neutral_diff_from_ref_mask

    current_multipliers = m_pos_pos + m_neg_neg + m_pos_neg + m_neg_pos + m_neutral
    return current_multipliers


def additional_dim_reverse_convolution(conv_transpose: nn.Module, previous_multipliers: torch.Tensor) -> \
        torch.Tensor:
    """
    calculates are reverse convolution step for a conv layer.

    special focus is on the dimensionality of the propagated multipliers regarding the output dimensionality of the
    full network. Since the reverse convolution layer expects shape (batch_size, in_channels, image_dim1, ...) and
    the multipliers are of shape (batch_size, output_dim, in_channels, image_dim1, ...), the multipliers are
    propagated for each output dimension separately and then concatenated to match the expected shape.

    Args:
        conv_transpose: transposed convolution that matches the parameters of the convolution layer
        previous_multipliers: multipliers of the previous layer to be propagated

    Returns:
        tensor with the multipliers propagated through the current convolution layer
    """
    multipliers = []
    for output_dim in range(previous_multipliers.shape[1]):
        if torch.cuda.is_available():
            previous_multipliers = previous_multipliers.cuda()
        current_multipliers = conv_transpose(previous_multipliers[:, output_dim])
        current_multipliers = current_multipliers.unsqueeze(1)
        multipliers.append(current_multipliers)
    result_multipliers = torch.cat(multipliers, dim=1)
    return result_multipliers


def linear_rule_conv(dl, current_layer: torch.jit._script.RecursiveScriptModule, current_layer_name: str,
                     previous_multipliers: torch.Tensor) -> torch.Tensor:
    """
    calculates the multipliers for the current conv layer using the linear rule

    Args:
        dl: DeepLiftClass object containing the model and the reference activations
        current_layer: layer for which the multipliers should be calculated
        current_layer_name: name of the layer for which the multipliers are calculated
        previous_multipliers: multipliers of the previous layer

    Returns:
        tensor with the multipliers for the current layer
    """
    if not hasattr(dl, "reference_activation"):
        raise AttributeError("reference activation was not set yet")
    layer_weights = current_layer.weight

    conv_transpose = set_up_conv_transpose_layer(current_layer=current_layer, layer_weights=layer_weights)
    if isinstance(previous_multipliers, Tuple):
        current_multipliers = linear_rule_conv_pos_neg(dl=dl, previous_multipliers=previous_multipliers,
                                                       current_layer=current_layer,
                                                       current_layer_name=current_layer_name,
                                                       reverse_conv=conv_transpose)
        return current_multipliers
    result_multipliers = additional_dim_reverse_convolution(conv_transpose, previous_multipliers)
    return result_multipliers
