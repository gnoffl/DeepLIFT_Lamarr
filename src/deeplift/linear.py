import torch
import src.deeplift.utils as utils
from typing import Tuple


def linear_rule_pos_neg(dl, current_layer: torch.jit._script.RecursiveScriptModule, current_layer_name: str,
                        previous_multipliers: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    calculates the multipliers for the current linear layer using the linear rule in case positive and negative
    multipliers are given separately

    Args:
        dl: DeepLiftObject that contains the model and the reference activations
        current_layer: layer for which the multipliers should be calculated
        current_layer_name: name of the layer for which the multipliers should be calculated
        previous_multipliers: multipliers of the previous layer in shape (A, B), where A are the positive and B the
            negative multipliers

    Returns:
        tensor with the multipliers for the current layer
    """
    pos_multipliers, neg_multipliers = previous_multipliers
    if torch.cuda.is_available():
        pos_multipliers = pos_multipliers.cuda()
        neg_multipliers = neg_multipliers.cuda()
    diff_from_ref_in = dl.diff_from_ref[dl.model.get_previous_layer_name(current_layer_name)]
    weights = current_layer.weight
    neg_diff_from_ref_mask, pos_diff_from_ref_mask, neutral_diff_from_ref_mask, pos_weight_mask, neg_weight_mask = utils.get_pos_neg_masks(
        weights=weights, diff_from_ref_in=diff_from_ref_in, for_shape=pos_multipliers)

    # calculation spelled out in all steps once for debugging purposes
    # positive multiplier is used for positive combination of weights and differences (++, --)
    pos_weights = weights * pos_weight_mask
    pos_pos_product = torch.matmul(pos_multipliers, pos_weights)
    m_pos_pos = pos_pos_product * pos_diff_from_ref_mask
    m_neg_neg = torch.matmul(pos_multipliers, weights * neg_weight_mask) * neg_diff_from_ref_mask
    # negative multiplier is used for negative combination of weights and differences (-+, +-)
    m_pos_neg = torch.matmul(neg_multipliers, weights * neg_weight_mask) * pos_diff_from_ref_mask
    m_neg_pos = torch.matmul(neg_multipliers, weights * pos_weight_mask) * neg_diff_from_ref_mask
    # special case of zero difference as defined in the paper
    m_neutral = torch.matmul((pos_multipliers + neg_multipliers) / 2, weights) * neutral_diff_from_ref_mask

    current_multipliers = m_pos_pos + m_neg_neg + m_pos_neg + m_neg_pos + m_neutral
    return current_multipliers


def linear_rule_linear(dl, current_layer_name: str,
                       previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    calculates the multipliers for the current linear layer using the linear rule

    Args:
        dl: DeepLiftObject that contains the model and the reference activations
        current_layer_name: name of the layer to calculate the multipliers for
        previous_multipliers: multipliers of the previous layer

    Returns:
        tensor with the multipliers for the current layer
    """
    if not hasattr(dl, "reference_activation"):
        raise AttributeError("reference activation was not set yet")
    current_layer = dl.model.get_layer(current_layer_name)
    if isinstance(previous_multipliers, torch.Tensor):
        if torch.cuda.is_available():
            previous_multipliers = previous_multipliers.cuda()
        weights_transposed = current_layer.weight
        current_multipliers = torch.matmul(previous_multipliers, weights_transposed)
    elif isinstance(previous_multipliers, tuple):
        current_multipliers = linear_rule_pos_neg(dl=dl, current_layer=current_layer,
                                                  current_layer_name=current_layer_name,
                                                  previous_multipliers=previous_multipliers)
    else:
        raise ValueError("previous multipliers must be single tensor(if reveal_cancel is not used) or tuple of"
                         "tensors! Received type was {}".format(type(previous_multipliers)))
    return current_multipliers
