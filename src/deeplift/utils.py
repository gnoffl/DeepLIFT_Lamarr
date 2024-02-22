import torch
import src.deeplift.constants as constants


def get_pos_neg_masks(diff_from_ref_in: torch.Tensor, weights: torch.Tensor, for_shape: torch.Tensor):
    """
    creates masks for the positive and negative diff_from_ref and weights.

    Args:
        diff_from_ref_in: difference from the reference activation of the input for the current layer
        weights: weights of the current layer
        for_shape: multiplier containing the shape for which the masks should be created (second dimension of the
            multiplier is the relevant one))

    Returns:
        masks for the positive and negative diff_from_ref and weights in a tuple (A, B, C, D, E) where A is the mask
        for the negative diff_from_ref, B is the mask for the negative weights, C is the mask for the neutral
        diff_from_ref, D is the mask for the positive diff_from_ref and E is the mask for the positive weights
    """
    pos_diff_from_ref_mask = diff_from_ref_in > constants.ZERO_MASK_THRESHOLD
    neg_diff_from_ref_mask = diff_from_ref_in < -constants.ZERO_MASK_THRESHOLD
    neutral_diff_from_ref_mask = torch.abs(diff_from_ref_in) <= constants.ZERO_MASK_THRESHOLD
    pos_weight_mask = weights > 0
    neg_weight_mask = weights < 0
    new_masks = []
    for mask in [pos_diff_from_ref_mask, neg_diff_from_ref_mask, neutral_diff_from_ref_mask]:
        new_masks.append(repeat_to_match_shapes(for_first_dim=for_shape, to_repeat=mask))
    pos_diff_from_ref_mask, neg_diff_from_ref_mask, neutral_diff_from_ref_mask = new_masks
    return (neg_diff_from_ref_mask, pos_diff_from_ref_mask, neutral_diff_from_ref_mask, pos_weight_mask,
            neg_weight_mask)


def repeat_to_match_shapes(for_first_dim: torch.Tensor, to_repeat: torch.Tensor) -> torch.Tensor:
    """
    repeats the tensor to_repeat to match the product of first dimension of the tensor for_first_dim and the first
    dimension of the tensor to_repeat

    Args:
        for_first_dim: tensor to repeat to match the first dimension of to_repeat
        to_repeat: tensor to repeat

    Returns:
        repeated tensor
    """
    to_repeat = to_repeat.unsqueeze(1)
    repeats = [1] * len(to_repeat.shape)
    repeats[1] = for_first_dim.shape[1]
    to_repeat = to_repeat.repeat(repeats)
    return to_repeat
