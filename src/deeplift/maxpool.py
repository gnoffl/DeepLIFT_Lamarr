import torch
import torch.nn as nn
from typing import Tuple

import deeplift.constants as constants


def get_max_unpool_layers(current_layer: torch.jit._script.RecursiveScriptModule) -> (
        Tuple[nn.MaxPool1d, nn.MaxUnpool1d] or Tuple[nn.MaxPool2d, nn.MaxUnpool2d] or
        Tuple[nn.MaxPool3d, nn.MaxUnpool3d]):
    """
    initializes the maxpool and maxunpool layers for the current maxpool layer

    Args:
        current_layer: the current maxpool layer

    Returns:
        tuple (A, B) of the maxpool layer A and maxunpool layer B
    """
    # take parameters from current layer and initialize other layer of same type with them
    kernel_size = current_layer.kernel_size
    stride = current_layer.stride
    padding = current_layer.padding
    dilation = current_layer.dilation
    # get names for layers
    layer_name = current_layer.original_name
    parts = layer_name.split("Pool")
    unpool_name = "Unpool".join(parts)
    # forward_layer not redundant with "current_layer", because the jit implementation does not return the indices
    forward_layer = getattr(nn, layer_name)(kernel_size=kernel_size, stride=stride, padding=padding,
                                            dilation=dilation, return_indices=True)
    reverse_layer = getattr(nn, unpool_name)(kernel_size=kernel_size, stride=stride, padding=padding)
    return forward_layer, reverse_layer


def maxpool_pos_neg(inputs: torch.Tensor, unpool_layer: torch.nn.Module, indexes: torch.Tensor,
                    previous_multipliers: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    calculates the multipliers for the current maxpool layer for positive and negative multipliers

    Args:
        inputs: inputs from the forward pass to the current maxpool layer
        unpool_layer: maxunpool layer that reverts the effects of the maxpool layer
        indexes: indices of the positions which were selected by the maxpool layer
        previous_multipliers: multipliers of the previous layer (one closer to the output) in shape (A, B), where
            A are the positive and B the negative multipliers

    Returns:
        multipliers for the current layer
    """
    pos_multipliers, neg_multipliers = previous_multipliers
    pos_input_mask = inputs > constants.ZERO_MASK_THRESHOLD
    neg_input_mask = inputs < -constants.ZERO_MASK_THRESHOLD
    zero_input_mask = torch.abs(inputs) <= constants.ZERO_MASK_THRESHOLD
    current_pos = max_unpool_extra_dim(indexes=indexes, input_values=inputs, multipliers=pos_multipliers,
                                       reverse_layer=unpool_layer)
    current_pos = current_pos * pos_input_mask
    current_neg = max_unpool_extra_dim(indexes=indexes, input_values=inputs, multipliers=neg_multipliers,
                                       reverse_layer=unpool_layer)
    current_neg = current_neg * neg_input_mask
    current_zero = max_unpool_extra_dim(indexes=indexes, input_values=inputs,
                                        multipliers=(pos_multipliers + neg_multipliers) / 2,
                                        reverse_layer=unpool_layer)
    current_zero = current_zero * zero_input_mask
    current_multipliers = current_pos + current_neg + current_zero
    return current_multipliers


def maxpool(dl, current_layer_name: str, previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) \
        -> torch.Tensor:
    """
    calculates the multipliers for the current maxpool layer

    Args:
        dl: DeepLiftClass object containing the model and relevant information
        current_layer_name: the name od the current maxpool layer
        previous_multipliers: multipliers of the previous layer

    Returns:
        multipliers for the current layer
    """
    if not hasattr(dl, "reference_activation"):
        raise AttributeError("reference activation was not set yet")

    current_layer = dl.model.get_layer(current_layer_name)
    input_values = dl.forward_activations[dl.model.get_previous_layer_name(current_layer_name)]
    forward_layer, reverse_layer = get_max_unpool_layers(current_layer)
    # get indices of the values that were passed forward by the maxpool layer
    _, indexes = forward_layer(input_values)
    if isinstance(previous_multipliers, tuple):
        current_multipliers = maxpool_pos_neg(unpool_layer=reverse_layer, indexes=indexes,
                                              previous_multipliers=previous_multipliers, inputs=input_values)
        return current_multipliers
    current_multipliers = max_unpool_extra_dim(indexes=indexes, input_values=input_values,
                                               multipliers=previous_multipliers, reverse_layer=reverse_layer)
    return current_multipliers


def max_unpool_extra_dim(indexes: torch.Tensor, input_values: torch.Tensor, multipliers: torch.Tensor,
                         reverse_layer: torch.nn.Module) -> torch.Tensor:
    """
    propagates multipliers through a maxpool layer

    Args:
        indexes: indexes that the maxpool layer used to propagate values in the forward pass
        input_values: inputs to the maxpool layer
        multipliers: multipliers to be propagated backwards
        reverse_layer: max unpooling layer that reverts the effects of the maxpool layer

    Returns:
        multipliers propagated backwards through the maxpool layer
    """
    to_concatenate = []
    for output_dim in range(multipliers.shape[1]):
        unpooled = reverse_layer(multipliers[:, output_dim], indexes,
                                 output_size=input_values.size())
        unpooled = unpooled.unsqueeze(1)
        to_concatenate.append(unpooled)
    current_multipliers = torch.cat(to_concatenate, dim=1)
    return current_multipliers
