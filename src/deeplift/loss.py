import torch
from typing import Tuple, Callable


def reverse_loss(dl, loss_function: str, target_values: torch.Tensor, non_linearity_method: Callable) \
        -> Tuple[torch.Tensor, str] or Tuple[Tuple[torch.Tensor, torch.Tensor], str]:
    """
    calculates multipliers for the given loss function

    Args:
        dl: DeepLiftClass object containing the model and the forward and reference activations
        loss_function: type of loss used
        target_values: target values for the output of the network. Needed for loss calculation.
        non_linearity_method: method to deal with non linearities

    Returns:
        calculated multipliers after the loss function as well as the name under which the results of the loss
        calculations are saved in the forward_activations and reference_activations
    """
    if loss_function.lower() in ["mean_squared_error", "mse"]:
        multipliers = mse_loss(dl=dl, target_output_values=target_values, non_linearity_method=non_linearity_method)
    else:
        raise ValueError(f"loss \"{loss_function}\" not supported!")
    return multipliers


def mse_loss(dl, target_output_values: torch.Tensor, non_linearity_method: Callable) \
        -> Tuple[torch.Tensor, str] or Tuple[Tuple[torch.Tensor, torch.Tensor], str]:
    """
    calculates the multipliers for the mean squared error loss

    Args:
        dl: DeepLiftClass object containing the model and the forward and reference activations
        target_output_values: target values for the output of the network. Needed for loss calculation.
        non_linearity_method: method to deal with non linearities

    Returns:
        calculated multipliers after the loss function as well as the name under which the results of the loss
        calculations are saved in the forward_activations and reference_activations
    """
    output_layer_name, _ = dl.model[-1]
    first_layer_name, _ = dl.model[0]
    reference_targets = target_output_values

    def squaring_layer(x: torch.Tensor) -> torch.Tensor:
        """
        layer that squares all input values

        Args:
            x: output to calculate the squares for.

        Returns:
            the calculated squares
        """
        return torch.pow(x, 2)

    # calculate reference loss
    output_layer_activation_reference = dl.reference_activation[output_layer_name]
    output_layer_deviance_reference = output_layer_activation_reference - reference_targets
    squared_reference = torch.pow(output_layer_deviance_reference, 2)
    dl.reference_activation["squared"] = squared_reference
    dl.reference_activation["mse"] = torch.mean(squared_reference)

    # calculate loss for given input
    output_layer_activation = dl.forward_activations[output_layer_name]
    output_layer_deviance = output_layer_activation - target_output_values
    squared_output = torch.pow(output_layer_deviance, 2)
    mse_loss_value = torch.mean(squared_output)

    dl.forward_activations["squared"] = squared_output
    dl.forward_activations["mse"] = mse_loss_value
    dl.diff_from_ref["squared"] = squared_output - squared_reference
    dl.diff_from_ref["mse"] = mse_loss_value - dl.reference_activation["mse"]

    # multipliers from averaging over the squares
    shape = list(output_layer_activation.shape)
    shape[0] = 1
    multipliers_for_averaging_layer = torch.full(size=shape, fill_value=1. / output_layer_activation.numel())
    dl.model._add_layer(layer_name="squared", layer=squaring_layer)
    multipliers_for_squared_layer = non_linearity_method(dl=dl, current_layer_name="squared",
                                                         previous_layer_name=output_layer_name,
                                                         previous_multipliers=multipliers_for_averaging_layer)
    dl.model._remove_layer(layer_name="squared")
    return multipliers_for_squared_layer, "mse"
