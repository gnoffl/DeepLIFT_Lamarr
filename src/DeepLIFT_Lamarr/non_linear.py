import torch
from typing import Tuple
import src.DeepLIFT_Lamarr.utils as utils
import src.DeepLIFT_Lamarr.constants as constants

elementwise_non_linear_layers = ["relu", "sigmoid"]


def rescale_rule_pos_neg(dl, current_layer_name: str, previous_multipliers: Tuple[torch.Tensor, torch.Tensor],
                         current_layer_multipliers: torch.Tensor) -> torch.Tensor:
    """
    propagates the multipliers through the current layer using the rescale rule for positive and negative
        multipliers

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: name of the current layer
        previous_multipliers: tuple of multipliers (A, B) to be propagated through the layer. A are the positive
            multipliers and B are the negative multipliers.
        current_layer_multipliers: multipliers for the current layer

    Returns:
        multipliers after propagation through the current layer
    """
    pos_multipliers, neg_multipliers = previous_multipliers
    if torch.cuda.is_available():
        pos_multipliers = pos_multipliers.cuda()
        neg_multipliers = neg_multipliers.cuda()
    # element wise multiplication of the previous multipliers with the current multipliers
    pos_propagated = pos_multipliers * current_layer_multipliers
    neg_propagated = neg_multipliers * current_layer_multipliers
    zero_propagated = ((pos_multipliers + neg_multipliers) / 2) * current_layer_multipliers
    result = torch.zeros_like(pos_propagated)
    # use masks of pos diff from ref of the current layer to determine which multipliers from the previous layer
    # are propagated backwards
    pos_diff_from_ref_mask = dl.diff_from_ref[current_layer_name] > constants.ZERO_MASK_THRESHOLD
    neg_diff_from_ref_mask = dl.diff_from_ref[current_layer_name] < -constants.ZERO_MASK_THRESHOLD
    zero_mask = torch.abs(dl.diff_from_ref[current_layer_name]) <= constants.ZERO_MASK_THRESHOLD
    result[pos_diff_from_ref_mask] = pos_propagated[pos_diff_from_ref_mask]
    result[neg_diff_from_ref_mask] = neg_propagated[neg_diff_from_ref_mask]
    result[zero_mask] = zero_propagated[zero_mask]
    return result


def rescale_multiplier_calculation(dl, current_layer_name: str, multiplier_shape_template: torch.Tensor,
                                   previous_layer_name: str = None) -> torch.Tensor:
    """
    calculates the multipliers for the current elementwise non-linear layer using the rescale rule

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: the name of the layerto calculate the multipliers for.
        previous_layer_name: name of the layer one step closer to the input. Name can also be inferred.
        multiplier_shape_template: tensor of the shape of the multipliers that are propagated.

    Returns:
        the multipliers for the current layer
    """
    if previous_layer_name is not None:
        preceding_layer = previous_layer_name
    else:
        preceding_layer = dl.model.get_previous_layer_name(current_layer_name)
    delta_x = dl.diff_from_ref[preceding_layer]
    delta_y = dl.diff_from_ref[current_layer_name]
    # component wise division of delta_y by delta_x
    current_multipliers = torch.div(delta_y, delta_x)
    # use gradient as multipliers if delta_x is too close to zero (to avoid instability)
    # (and delta_y / delta_x approaches the gradient from very small values of delta_x)
    mask = torch.abs(delta_x) < constants.NUMERICAL_INSTABILITY_THRESHOLD
    if torch.any(mask):
        gradient = calculate_gradient(dl=dl, current_layer=current_layer_name, preceding_layer=preceding_layer)
        current_multipliers[mask] = gradient[mask]
    current_multipliers = utils.repeat_to_match_shapes(to_repeat=current_multipliers,
                                                       for_first_dim=multiplier_shape_template)
    return current_multipliers


def rescale_rule(dl, current_layer_name: str,
                 previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor],
                 previous_layer_name: str = None) -> torch.Tensor:
    """
    calculates the multipliers for the current layer using the rescale rule

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: layer for which the multipliers should be calculated
        previous_multipliers: multipliers of the previous layer
        previous_layer_name: name of the layer one step closer to the input. Name can also be inferred, so it does
            not need to be provided.

    Returns:
        multipliers for the current layer
    """
    if not hasattr(dl, "reference_activation"):
        raise AttributeError("reference activation was not set yet")
    if isinstance(previous_multipliers, tuple):
        is_tuple = True
        for_shape = previous_multipliers[0]
    elif isinstance(previous_multipliers, torch.Tensor):
        is_tuple = False
        for_shape = previous_multipliers
    else:
        raise ValueError("Previous multipliers must be single tensor (if rescale is used) or tuple of"
                         "tensors (if reveal_cancel was used)!\n"
                         "Received type was {}".format(type(previous_multipliers)))
    current_multipliers = rescale_multiplier_calculation(dl=dl, current_layer_name=current_layer_name,
                                                         previous_layer_name=previous_layer_name,
                                                         multiplier_shape_template=for_shape)
    if is_tuple:
        return rescale_rule_pos_neg(dl=dl, current_layer_name=current_layer_name,
                                    previous_multipliers=previous_multipliers,
                                    current_layer_multipliers=current_multipliers)
    # element wise multiplication of the previous multipliers with the current multipliers
    if torch.cuda.is_available():
        previous_multipliers = previous_multipliers.cuda()
    propagated_multipliers = previous_multipliers * current_multipliers
    return propagated_multipliers


def calculate_gradient(dl, current_layer: str, preceding_layer: str) -> torch.Tensor:
    """
    calculates the gradient of the values in the current layer with respect to the values in the preceding layer

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer: name of the current layer
        preceding_layer: name of the preceding layer (lay one closer to the input layer)

    Returns:
        gradient of the current layer with respect to the preceding layer
    """
    preceding_layer_activation = dl.forward_activations[preceding_layer]
    current_layer_activation = dl.forward_activations[current_layer]
    gradient = get_gradient(dl=dl, outputs=current_layer_activation, inputs=preceding_layer_activation,
                            grad_outputs=torch.ones_like(current_layer_activation))
    return gradient[0]


def get_pos_neg_contributions_non_linear(dl, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the positive and negative contributions a elementwise non-linear layer has on its following layer.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        layer_name: name of the layer to calculate the contributions for

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions of the current
        layer on the subsequent layer.
    """
    # since the layers operate elementwise, each output value is only dependent on the corresponding input value.
    # therefore, a single value can only have positive or negative contributions, but not both. The positive
    # and negative contributions are just the positive and negative values from the output separated.
    diff_from_ref = dl.diff_from_ref[layer_name]
    pos_diff_from_ref = diff_from_ref * (diff_from_ref > 0)
    neg_diff_from_ref = diff_from_ref * (diff_from_ref < 0)
    return pos_diff_from_ref, neg_diff_from_ref


def get_pos_neg_contributions_maxpool(dl, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the positive and negative contributions a maxpool layer has on its following layer.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        layer_name: name of the current layer

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions of the current
        layer on the subsequent layer.
    """
    # since every output value of the maxpool layer is just a value from the input, it is not possible to have
    # positive and negative contributions to the same output value. Therefore, the positive and negative
    # contributions are just the positive and negative values from the output separated.
    outputs = dl.diff_from_ref[layer_name]
    pos_outputs = outputs * (outputs > 0)
    neg_outputs = outputs * (outputs < 0)
    return pos_outputs, neg_outputs


def get_pos_neg_contributions_avgpool(dl, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the positive and negative contributions an avgpool layer has on its following layer.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        layer_name: name of the current layer

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions of the current
        layer on the subsequent layer.
    """
    current_layer = dl.model.get_layer(layer_name)
    previous_layer_name = dl.model.get_previous_layer_name(layer_name)
    input_values = dl.forward_activations[previous_layer_name]
    pos_inputs = input_values * (input_values > 0)
    neg_inputs = input_values * (input_values < 0)
    # since all weights in avg pool are positive, no distinction needs to be made for them
    pos_contributions = current_layer(pos_inputs)
    neg_contributions = current_layer(neg_inputs)
    return pos_contributions, neg_contributions


def get_pos_neg_contributions_conv(dl, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the positive and negative contributions a conv layer has on its following layer.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        layer_name: name of the layer to calculate the contributions for.

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions.
    """
    current_layer = dl.model.get_layer(layer_name)
    original_layer_weights = current_layer.weight
    original_layer_bias = current_layer.bias
    positive_weights = original_layer_weights * (original_layer_weights > 0)
    negative_weights = original_layer_weights * (original_layer_weights < 0)

    preceding_layer_name = dl.model.get_previous_layer_name(layer_name)
    input_values = dl.diff_from_ref[preceding_layer_name]
    pos_inputs = input_values * (input_values > 0)
    neg_inputs = input_values * (input_values < 0)

    # calculate contributions with positive weights
    current_layer.weight = positive_weights
    pos_contributions = current_layer(pos_inputs)
    neg_contributions = current_layer(neg_inputs)

    # calculate contributions with negative weights
    current_layer.weight = negative_weights
    pos_contributions += current_layer(neg_inputs)
    neg_contributions += current_layer(pos_inputs)

    # remove bias from contributions
    if original_layer_bias is not None:
        multiplicator = original_layer_bias
        for _ in range(len(original_layer_weights.shape) - 2):
            multiplicator = multiplicator.unsqueeze(1)
        pos_contributions -= 2 * multiplicator
        neg_contributions -= 2 * multiplicator

    # set weights of the current layer back to its original values
    current_layer.weight = original_layer_weights

    return pos_contributions, neg_contributions


def get_pos_neg_contributions_input(dl):
    """
    calculates the positive and negative contributions the input layer has on its following layer.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions.
    """
    input_values = dl.diff_from_ref["input"]
    pos_input_mask = input_values > 0
    neg_input_mask = input_values < 0
    pos_contributions = input_values * pos_input_mask
    neg_contributions = input_values * neg_input_mask
    return pos_contributions, neg_contributions


def get_pos_neg_contributions_linear(dl, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the positive and negative contributions a linear layer has on its following layer.

    Important for the reveal cancel rule as delta_x_plus and delta_x_minus.
    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        layer_name: name the linear layer for which the output contributions should be calculated.

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions.
    """
    preceding_layer_name = dl.model.get_previous_layer_name(layer_name)
    input_values = dl.diff_from_ref[preceding_layer_name]
    pos_input_mask = input_values > 0
    neg_input_mask = input_values < 0
    # if the current layer is the input layer, the contributions are just the input values
    # todo: test this case
    if layer_name == "input":
        pos_contributions = input_values * pos_input_mask
        neg_contributions = input_values * neg_input_mask
        return pos_contributions, neg_contributions
    weights = dl.model.get_layer(layer_name).weight
    pos_weight_mask = weights > 0
    neg_weight_mask = weights < 0
    pos_contributions = torch.matmul(input_values * pos_input_mask,
                                     torch.transpose(weights * pos_weight_mask, 0, 1)) + \
                        torch.matmul(input_values * neg_input_mask,
                                     torch.transpose(weights * neg_weight_mask, 0, 1))
    neg_contributions = torch.matmul(input_values * pos_input_mask,
                                     torch.transpose(weights * neg_weight_mask, 0, 1)) + \
                        torch.matmul(input_values * neg_input_mask,
                                     torch.transpose(weights * pos_weight_mask, 0, 1))
    return pos_contributions, neg_contributions


def calculate_reveal_cancel_contributions(dl, current_layer_name: str, main_input_delta: torch.Tensor,
                                          secondary_input_delta: torch.Tensor) -> torch.Tensor:
    """
    calculates the influence of the different inputs for the output for the reveal cancel rule.

    for the calculation of delta_y_plus and delta_y_minus from the deeplift paper.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: name of the current non-linear layer
        main_input_delta: delta_x_plus for the calculation of delta_y_plus or delta_x_minus for the calculation of
            delta_y_minus
        secondary_input_delta: delta_x_minus for the calculation of delta_y_plus or delta_x_plus for the
            calculation of delta_y_minus

    Returns:
        influence of the different inputs for the output, corresponding to delta_y_plus or delta_y_minus.

    """
    layer_object = dl.model.get_layer(current_layer_name)
    previous_layer_name = dl.model.get_previous_layer_name(current_layer_name)
    reference_input = dl.reference_activation[previous_layer_name]
    reference_output_standard = dl.reference_activation[current_layer_name]
    influence_standard = layer_object(reference_input + main_input_delta)
    reference_output_adjusted = layer_object(reference_input + secondary_input_delta)
    influence_adjusted = layer_object(reference_input + main_input_delta + secondary_input_delta)
    standard_diff = influence_standard - reference_output_standard
    adjusted_diff = influence_adjusted - reference_output_adjusted
    result = (standard_diff + adjusted_diff) / 2
    return result


def get_gradient(dl, outputs: torch.Tensor, inputs: torch.Tensor, grad_outputs: torch.Tensor) -> torch.Tensor:
    """
    calculates the gradient of the outputs with respect to the inputs.

    The gradient is cached in the DeepLiftClass object, so it is only calculated once. Multiple calculations of the same
        gradient lead to an error.

    Args:
        dl: DeepLiftClass object containing the model and other information, including a gradient cache
        outputs: output tensor to calculate the gradient for
        inputs: input tensor to calculate the gradient for
        grad_outputs: grad_outputs argument for torch.autograd.grad

    Returns:
        gradient of the outputs with respect to the inputs
    """
    gradient = dl.gradients.get((outputs, inputs))
    if gradient is None:
        gradient = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs)
        dl.gradients[(outputs, inputs)] = gradient
    return gradient


def reveal_cancel_exception(dl, current_layer_name: str, other_delta: torch.Tensor,
                            x0_gradient: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    reveal cancel rule for the special case of very small delta_x.

    Since the reveal_cancel rule divides by delta_x, numerical instabilities might arise. In this case, the gradient
    is used instead.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: name of the current non-linear layer to propagate the multipliers through
        other_delta: The delta_x that is used to determine the position of the gradient, but not used in the
            gradient calculation.
        x0_gradient: gradient of the current layer at the position of the reference input. Will be calculated if
            not provided. Needs to be given if it was calculated before, because autograd throws an error if the
            gradient is calculated twice.

    Returns:
        multipliers for the current layer in all positions, not only where delta_x is too small.
    """
    current_layer = dl.model.get_layer(current_layer_name)
    previous_layer_name = dl.model.get_previous_layer_name(current_layer_name)
    reference_input = dl.reference_activation[previous_layer_name]
    if x0_gradient is None:
        reference_output = dl.reference_activation[current_layer_name]
        gradient1 = get_gradient(dl=dl, outputs=reference_output, inputs=reference_input,
                                 grad_outputs=torch.ones_like(reference_output))
    else:
        gradient1 = x0_gradient
    # if other_delta is zero, the gradient is used again. Important, because otherwise the gradient would be
    # calculated twice, which throws an error
    if torch.all(torch.abs(other_delta) == 0):
        gradient2 = gradient1
    else:
        second_position = reference_input + other_delta
        second_output = current_layer(second_position)
        gradient2 = get_gradient(dl=dl, outputs=second_output, inputs=second_position,
                                 grad_outputs=torch.ones_like(second_output))
    average = (gradient1[0] + gradient2[0]) / 2
    return average, gradient1


def adjust_multipliers_by_gradient(dl, current_layer: str, del_x_minus: torch.Tensor, del_x_plus: torch.Tensor,
                                   multipliers_minus: torch.Tensor, multipliers_plus: torch.Tensor) -> None:
    """
    calculates the gradient for the reveal cancel rule if delta_x is too small.

    multipliers_plus and multipliers_minus are adjusted to contain the gradient in case delta_x is too small.

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer: name of the current layer
        del_x_minus: negative contributions to the input of the current layer
        del_x_plus: positive contributions to the input of the current layer
        multipliers_minus: negative multipliers calculated without the gradient
        multipliers_plus: positive multipliers calculated without the gradient
    """
    mask_del_x_plus = torch.abs(del_x_plus) < constants.NUMERICAL_INSTABILITY_THRESHOLD
    if torch.any(mask_del_x_plus):
        multipliers_plus_exception, x0_gradient = reveal_cancel_exception(dl=dl, current_layer_name=current_layer,
                                                                          other_delta=del_x_minus)
        multipliers_plus[mask_del_x_plus] = multipliers_plus_exception[mask_del_x_plus]
    mask_del_x_minus = torch.abs(del_x_minus) < constants.NUMERICAL_INSTABILITY_THRESHOLD
    if torch.any(mask_del_x_minus):
        # important that x0_gradient is used for reveal_cancel_exception, because otherwise the gradient would be
        # calculated twice, which throws an error
        x0_gradient = x0_gradient if torch.any(mask_del_x_plus) else None
        multipliers_minus_exception = reveal_cancel_exception(dl=dl, current_layer_name=current_layer,
                                                              other_delta=del_x_plus,
                                                              x0_gradient=x0_gradient)[0]
        multipliers_minus[mask_del_x_minus] = multipliers_minus_exception[mask_del_x_minus]


def get_pos_neg_contributions(dl, previous_layer_name: str, current_layer_name: str)\
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    gets the positive and negative contributions the previous layer has on the current layer

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        previous_layer_name: name of the layer one step closer to the input layer
        current_layer_name: name of the current layer

    Returns:
        Tuple (A, B) where A are the positive contributions and B are the negative contributions from the previous
            layer on the current layer.
    """
    if previous_layer_name is None:
        previous_layer_name = dl.model.get_previous_layer_name(current_layer_name)

    if previous_layer_name == "input":
        return get_pos_neg_contributions_input(dl=dl)

    previous_layer = dl.model.get_layer(previous_layer_name)
    if previous_layer.original_name.lower() == "linear":
        return get_pos_neg_contributions_linear(dl=dl, layer_name=previous_layer_name)

    if previous_layer.original_name.lower().startswith("conv"):
        return get_pos_neg_contributions_conv(dl=dl, layer_name=previous_layer_name)

    if previous_layer.original_name.lower().startswith("avgpool"):
        return get_pos_neg_contributions_avgpool(dl=dl, layer_name=previous_layer_name)

    if previous_layer.original_name.lower().startswith("maxpool"):
        return get_pos_neg_contributions_maxpool(dl=dl, layer_name=previous_layer_name)
    if previous_layer.original_name.lower() in elementwise_non_linear_layers:
        return get_pos_neg_contributions_non_linear(dl=dl, layer_name=previous_layer_name)

    if previous_layer.original_name.lower().startswith("flatten"):
        preprevious_layer_name = dl.model.get_previous_layer_name(previous_layer_name)
        del_x_plus, del_x_minus = get_pos_neg_contributions(dl=dl, previous_layer_name=preprevious_layer_name,
                                                            current_layer_name=previous_layer_name)
        del_x_plus = dl.model.get_layer(previous_layer_name)(del_x_plus)
        del_x_minus = dl.model.get_layer(previous_layer_name)(del_x_minus)
        return del_x_plus, del_x_minus
    # todo: update list of accepted layers and error message!
    raise ValueError(f"layer must be linear or conv! Given was layer name \"{previous_layer_name}\" with original"
                     f"name \"{previous_layer.original_name}\"")


def reveal_cancel_rule(dl, current_layer_name: str,
                       previous_multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor],
                       previous_layer_name: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculates the multipliers for the current non-linear layer using the reveal cancel rule

    Args:
        dl: DeepLiftClass object containing the model and the reference and forward activations
        current_layer_name: name of the layer to calculate the multipliers for
        previous_multipliers: multipliers from the previous layer, which need to be propagated
        previous_layer_name: name of the layer one step closer to the input. Name can also be inferred, so it
            usually does not need to be provided.

    Returns:
        Tuple (A, B) where A are the multipliers for the positive contributions and B are the multipliers for the
        negative contributions.
    """
    # set up multiplication for later use while also error checking
    if isinstance(previous_multipliers, torch.Tensor):
        old_pos = previous_multipliers
        old_neg = previous_multipliers
    elif isinstance(previous_multipliers, tuple):
        old_pos, old_neg = previous_multipliers
    else:
        raise ValueError("Previous multipliers must be single tensor (if rescale is used) or tuple of"
                         "tensors (if reveal_cancel was used)!\n"
                         "Received type was {}".format(type(previous_multipliers)))
    del_x_plus, del_x_minus = get_pos_neg_contributions(dl=dl, previous_layer_name=previous_layer_name,
                                                        current_layer_name=current_layer_name)
    del_y_plus = calculate_reveal_cancel_contributions(dl=dl, current_layer_name=current_layer_name,
                                                       main_input_delta=del_x_plus,
                                                       secondary_input_delta=del_x_minus)
    del_y_minus = calculate_reveal_cancel_contributions(dl=dl, current_layer_name=current_layer_name,
                                                        main_input_delta=del_x_minus,
                                                        secondary_input_delta=del_x_plus)
    multipliers_plus = del_y_plus / del_x_plus
    multipliers_minus = del_y_minus / del_x_minus
    # numerical instabilities arise when delta_x gets too small. In this case, the gradient is used instead.
    adjust_multipliers_by_gradient(dl=dl, current_layer=current_layer_name, del_x_minus=del_x_minus,
                                   del_x_plus=del_x_plus, multipliers_minus=multipliers_minus,
                                   multipliers_plus=multipliers_plus)
    current_multipliers_plus = utils.repeat_to_match_shapes(to_repeat=multipliers_plus,
                                                            for_first_dim=old_pos)
    current_multipliers_minus = utils.repeat_to_match_shapes(to_repeat=multipliers_minus,
                                                             for_first_dim=old_pos)
    # element wise multiplication of the previous multipliers with the current multipliers
    if torch.cuda.is_available():
        old_pos = old_pos.cuda()
        old_neg = old_neg.cuda()
    propagated_multipliers_plus = old_pos * current_multipliers_plus
    propagated_multipliers_minus = old_neg * current_multipliers_minus
    return propagated_multipliers_plus, propagated_multipliers_minus
