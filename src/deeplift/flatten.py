import torch
from typing import Tuple, List


def flatten_reverse(dl, shape: List[int], previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) \
        -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
    """
    reverts the flattening of the multipliers

    Args:
        dl: DeepLiftClass object containing the model and relevant information
        shape: shape of the layer
        previous_multipliers: multipliers from the previous layer to be propagated or tuple (A, B) of the positive
            (A) and negative (B) multipliers from the previous layer to be propagated

    Returns:
        multipliers in the shape of the input of the flatten layer. If the previous multipliers were a tuple, the
        returned multipliers are also a tuple of the same shape
    """
    if isinstance(previous_multipliers, torch.Tensor):
        output_dims = previous_multipliers.shape[1]
        shape = list(shape)
        shape.insert(1, output_dims)
        current_multipliers = previous_multipliers.view(shape)
        return current_multipliers
    elif isinstance(previous_multipliers, tuple):
        pos_multipliers, neg_multipliers = previous_multipliers
        pos_multipliers_shaped = flatten_reverse(dl=dl, shape=shape, previous_multipliers=pos_multipliers)
        neg_multipliers_shaped = flatten_reverse(dl=dl, shape=shape, previous_multipliers=neg_multipliers)
        return pos_multipliers_shaped, neg_multipliers_shaped
