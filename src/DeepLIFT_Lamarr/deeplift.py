from typing import Tuple, Dict, Callable, List
import torch
import warnings
import src.DeepLIFT_Lamarr.parsing as ps
import src.DeepLIFT_Lamarr.non_linear as non_linear
import src.DeepLIFT_Lamarr.utils as utils
import src.DeepLIFT_Lamarr.flatten as flatten
import src.DeepLIFT_Lamarr.maxpool as maxpool
import src.DeepLIFT_Lamarr.linear as linear
import src.DeepLIFT_Lamarr.convolution as convolution
import src.DeepLIFT_Lamarr.loss as loss
import src.DeepLIFT_Lamarr.avgpool as avgpool


class DeepLiftClass:
    model: ps.LoadedModel
    reference_activation: Dict[str, torch.Tensor] or None
    reference_activation_lists: Dict[str, List[torch.Tensor]] or None
    gradients: Dict[Tuple, torch.Tensor] or None
    diff_from_ref: Dict[str, torch.Tensor] or None
    forward_activations: Dict[str, torch.Tensor] or None

    def __init__(self, model: torch.jit._script.RecursiveScriptModule or ps.LoadedModel, reference_value: torch.Tensor,
                 shap=True) -> None:
        """
        initializes the DeepLiftClass and sets the model and corresponding reference activations

        Args:
            model: model for which explanations should be calculated
            reference_value: baseline value for the input. All explanations will be relative to this value and its
                activations / outputs.
            shap: determines whether shap values are calculated, or the standard Deeplift method is used. Makes a
                difference in the calculation of the reference values.
        """
        if isinstance(model, torch.jit._script.RecursiveScriptModule):
            model = ps.SequentialLoadedModel(model)
        elif not isinstance(model, ps.LoadedModel):
            raise ValueError("model must be of type LoadedModel or RecusiveScriptModule!")
        self.model = model

        if torch.cuda.is_available():
            self.model.model = self.model.model.cuda()
            reference_value = reference_value.cuda()

        if shap:
            self.set_reference_activation_multiple(reference_value)
        else:
            self.set_reference_activation_single(reference_value)
        self.gradients = {}

    def set_forward_activations(self, x: torch.Tensor) -> None:
        """
        does a forward pass of the input and saves the activations of each layer in a dict as an attribute.

        Args:
            x: the input
        """
        if hasattr(self, "forward_activations") and self.forward_activations is not None:
            raise AttributeError("forward activations already set!")
        if torch.cuda.is_available():
            x = x.cuda()
        activations = self.calculate_forward_activations(x)
        self.forward_activations = activations

    def calculate_forward_activations(self, x: torch.Tensor) -> Dict:
        """
        does a forward pass of the input and saves the activations of each layer in a dict as an attribute.

        Also returns the activations for further use.

        Args:
            x: the input

        Returns:
            dict with input under input and the activation after each layer under the layer layer_name
        """
        if torch.cuda.is_available():
            x = x.cuda()
        activations = {"input": x}
        for layer_name, layer in self.model:
            if torch.cuda.is_available():
                layer = layer.cuda()
            x = layer(x)
            activations[layer_name] = x
        return activations

    def set_reference_activation_single(self, reference_input: torch.Tensor) -> None:
        """
        sets the reference activation for a single input value

        Args:
            reference_input: value to set the reference activation for
        """
        if hasattr(self, "reference_activation") and self.reference_activation is not None:
            raise AttributeError("reference activation already set!")
        if reference_input.shape[0] != 1:
            warnings.warn("batch dimension of reference input is not 1! If you want to average over multiple inputs,"
                          " use shap=True for initialization!")

        activations = self.calculate_forward_activations(reference_input)
        self.reference_activation = activations

    def set_reference_activation_multiple(self, reference_dataset: torch.Tensor) -> None:
        """
        sets the reference activation by averaging the activation for each layer for each input.

        Args:
            reference_dataset: the dataset to calculate the reference activation with. Need to be a tensor of shape
                batch_size x input_shape.
        """
        if hasattr(self, "reference_activation") and self.reference_activation is not None:
            raise AttributeError("reference activation already set!")
        if reference_dataset.shape[0] == 1:
            warnings.warn("reference dataset only contains one input. This is not recommended!")

        activations = {}
        for layer_name, layer in self.model:
            activations[layer_name] = []
        activations["input"] = []
        for i in range(len(reference_dataset)):
            batch = reference_dataset[i:i + 1]
            if torch.cuda.is_available():
                batch = batch.cuda()
            batch_activations = self.calculate_forward_activations(batch)
            for layer_name, layer_activation in batch_activations.items():
                activations[layer_name].append(layer_activation)
        self.reference_activation_lists = activations

    def set_diff_from_ref(self, input_tens: torch.Tensor) -> None:
        """
        calculates the difference between the forward activations of the input and the reference activation

        Args:
            input_tens: input values to calculate the difference for

        Returns:
            tensor with the difference between the forward activations and the reference activation
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet!")
        if not hasattr(self, "forward_activations"):
            self.set_forward_activations(input_tens)
        if not self.forward_activations.keys() == self.reference_activation.keys():
            raise ValueError("keys for forward activation and reference activation dont match!")
        differences = {}
        for (layer_name, reference_val) in self.reference_activation.items():
            if torch.cuda.is_available():
                self.forward_activations[layer_name] = self.forward_activations[layer_name].cuda()
            differences[layer_name] = self.forward_activations[layer_name].sub(reference_val)
        self.diff_from_ref = differences

    def get_deltas(self, contributions: torch.Tensor, output_layer_of_interest: str, target_output_index: int = None)\
            -> torch.Tensor:
        """
        calculates the difference between the summed up explanations and the observed outputs

        Args:
            contributions: contributions of every input for every output in each batch. (shape: batch_size x n(outputs)
                x shape(inputs))
            output_layer_of_interest: layer for which the contributions and deltas are calculated
            target_output_index: target class for which the contributions should be calculated as index of the output
                vector

        Returns:
            Tensor of shape batch_size x n(outputs) containing the difference between the summed up explanations and the
            observed outputs for each output in each batch
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet!")
        summed_contributions = torch.sum(contributions, dim=tuple(range(2, len(contributions.shape))))
        reference_output = self.reference_activation[output_layer_of_interest]
        if target_output_index:
            reference_output = reference_output[:, target_output_index]
        predictions = torch.add(summed_contributions, reference_output)
        output = self.forward_activations[output_layer_of_interest]
        if target_output_index:
            output = output[:, target_output_index:target_output_index + 1]
        deltas = torch.sub(predictions, output)
        return deltas

    def calculate_contributions(self, input_layer_of_interest: str,
                                multipliers: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        calculates the contributions of the input to the output of the layer of interest.

        Args:
            input_layer_of_interest: name of the input layer to calculate the contributions for
            multipliers: the multipliers for the input layer

        Returns:
            the contributions of the input to the output of the layer of interest
        """
        # multiply the multipliers with the difference from the reference activation to get contributions
        diff_layer = self.model.get_previous_layer_name(input_layer_of_interest)
        if isinstance(multipliers, torch.Tensor):
            differences = self.diff_from_ref[diff_layer]
            # adjust shape of differences to match multipliers (batch x output features x input shape)
            for_shape = multipliers
            differences = utils.repeat_to_match_shapes(to_repeat=differences, for_first_dim=for_shape)
            contributions = torch.mul(multipliers, differences)
        elif isinstance(multipliers, tuple):
            for_shape = multipliers[0]
            pos_contribs, neg_contribs = non_linear.get_pos_neg_contributions(
                dl=self, previous_layer_name=diff_layer, current_layer_name=input_layer_of_interest
            )

            pos_contribs = utils.repeat_to_match_shapes(to_repeat=pos_contribs, for_first_dim=for_shape)
            neg_contribs = utils.repeat_to_match_shapes(to_repeat=neg_contribs, for_first_dim=for_shape)
            pos_multipliers, neg_multipliers = multipliers
            pos_contributions = torch.mul(pos_multipliers, pos_contribs)
            neg_contributions = torch.mul(neg_multipliers, neg_contribs)
            contributions = pos_contributions + neg_contributions
        else:
            raise ValueError("multipliers must be single tensor (if rescale is used) or tuple of"
                             "tensors (if reveal_cancel was used)!\n"
                             "Received type was {}".format(type(multipliers)))
        return contributions

    def apply_rule(self, layer: torch.jit._script.RecursiveScriptModule, layer_name: str, multipliers: torch.Tensor,
                   non_linearity_method: Callable) -> torch.Tensor:
        """
        applies one rule for calculating the multipliers for the current layer

        Args:
            layer: current layer
            layer_name: name of the current layer
            multipliers: current state of the multipliers being back propagated
            non_linearity_method: method to deal with non linearities

        Returns:
            multipliers for the current layer
        """
        layer_type = layer.original_name
        previous_layer_name = self.model.get_previous_layer_name(layer_name)

        if layer_type == "Linear":
            multipliers = linear.linear_rule_linear(dl=self, current_layer_name=layer_name,
                                                    previous_multipliers=multipliers)
        elif layer_type.startswith("Conv"):
            multipliers = convolution.linear_rule_conv(dl=self, current_layer=layer, previous_multipliers=multipliers,
                                                       current_layer_name=layer_name)
        elif layer_type.startswith("MaxPool"):
            multipliers = maxpool.maxpool(dl=self, current_layer_name=layer_name, previous_multipliers=multipliers)
        elif layer_type.startswith("AvgPool"):
            output_from_prev_layer = self.diff_from_ref[previous_layer_name]
            multipliers = avgpool.avgpool(dl=self, current_layer=layer, prev_layer_diff=output_from_prev_layer,
                                          previous_multipliers=multipliers)
        elif layer_type == "Flatten":
            diff_from_ref_prev_layer = self.diff_from_ref[previous_layer_name]
            multipliers = flatten.flatten_reverse(dl=self, previous_multipliers=multipliers,
                                                  shape=diff_from_ref_prev_layer.shape)
        elif layer_type.lower() in non_linear.elementwise_non_linear_layers:
            multipliers = non_linearity_method(dl=self, current_layer_name=layer_name, previous_multipliers=multipliers)
        elif layer_type == "Dropout":
            pass
        else:
            raise ValueError(f"layer type \"{layer_type}\" not supported!")
        return multipliers

    def set_initial_multiplier(self, output_layer_of_interest: str) -> torch.Tensor:
        """
        sets the initial multipliers for the output layer of interest, adjusted for the correct shape of batchsize x
        output features x output features

        Multipliers are given the shape batchsize x output features x output features. This means that after back
        propagation, the multipliers for the first layer are already in the correct shape (batchsize x output features x
        input features). This for now only works for layers with one dimensional output.

        Args:
            output_layer_of_interest: output layer to calculate the contributions for. Initial multipliers are need to
                be adjusted to the shape of the out of this layer. Layer must have one dimensional output.

        Returns:
            the initial multipliers
        """
        shape = self.forward_activations[output_layer_of_interest].shape
        if len(shape) > 2:
            raise ValueError("output layer of interest must have one dimensional output! This could be linear layers, "
                             "ReLU or other activations of linear layers or anything after a flatten layer.")
        batch_size = shape[0]
        current_layer_output_length = shape[1]
        multipliers = torch.eye(current_layer_output_length)
        multipliers.unsqueeze_(0)
        repeats = [1] * len(multipliers.shape)
        repeats[0] = batch_size
        multipliers = multipliers.repeat(repeats)
        return multipliers

    def check_and_convert_input_attributes(self, input_layer_of_interest: str or int,
                                           output_layer_of_interest: str or int, target_output_index: int,
                                           target_output_values: torch.Tensor, non_linearity_method: str) \
            -> Tuple[str, str, Callable]:
        """
        checks the input for the attribute method

        Args:
            input_layer_of_interest: input layer for which the contributions should be calculated
            output_layer_of_interest: output layer for which the contributions should be calculated
            target_output_index: target class for which the contributions should be calculated as index of the output
                vector
            target_output_values: target output values to calculate losses. If this parameter is given, the
                contributions will be calculated for the loss
            non_linearity_method: method to use for calculating the multipliers at non-linear layers

        Returns:
            Tuple of (layer_of_interest converted to string and method to use for calculating the multipliers at
            non-linear layers)
        """
        input_layer_of_interest = 0 if input_layer_of_interest is None else input_layer_of_interest
        if isinstance(input_layer_of_interest, int):
            input_layer_of_interest = self.model[input_layer_of_interest][0]
        if input_layer_of_interest not in self.model.get_all_layer_names():
            raise ValueError("input layer of interest not found in model!")

        output_layer_of_interest = -1 if output_layer_of_interest is None else output_layer_of_interest
        if isinstance(output_layer_of_interest, int):
            output_layer_of_interest = self.model[output_layer_of_interest][0]
        if output_layer_of_interest not in self.model.get_all_layer_names():
            raise ValueError("output layer of interest not found in model!")

        if target_output_values is not None and target_output_index is not None:
            raise ValueError("target and loss cannot be set at the same time!")
        if not hasattr(self, "reference_activation") and not hasattr(self, "reference_activation_lists"):
            raise AttributeError("reference activation was not set yet!")

        if non_linearity_method.upper() == "RESCALE":
            non_linearity_method = non_linear.rescale_rule
        elif non_linearity_method.upper() in ["REVEAL_CANCEL", "REVEAL-CANCEL", "REVEALCANCEL"]:
            non_linearity_method = non_linear.reveal_cancel_rule
        else:
            raise ValueError("method not supported!")

        return input_layer_of_interest, output_layer_of_interest, non_linearity_method

    def attribute(self, input_tensor: torch.Tensor, input_layer_of_interest: str or int = None,
                  output_layer_of_interest: str or int = None, target_output_index: int = None,
                  non_linearity_method: str = "rescale", loss_function: str = "mse",
                  target_output_values: torch.Tensor = None, deltas_per_baseline: bool = False)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculates the contributions of the input to the output of the layer of interest

        Args:
            input_tensor: values for which the contributions should be calculated
            input_layer_of_interest: input layer for which the contributions should be calculated
            output_layer_of_interest: output layer for which the contributions should be calculated
            target_output_index: target class for which the contributions should be calculated as index of the output
                vector
            non_linearity_method: method for calculating the multipliers
            loss_function: name of loss function to be used to explain loss. target values need to be given to
                calculate contributions for the loss
            target_output_values: target output values to calculate losses. If this parameter is given, the
                contributions will be calculated relative to the loss
            deltas_per_baseline: determines whether the deltas should be calculated for each baseline separately and
                added at the end, or if the deltas should be calculated for the average of the contributions.

        Returns: a Tuple (contribs, deltas) containing the contributions of the input layer of interest  to the output
        of the output layer of interest as contribs and the difference between the summed up contributions and the
        observed output as deltas.
        """

        input_layer_of_interest, output_layer_of_interest, non_linearity_method = self.check_and_convert_input_attributes(
            input_layer_of_interest=input_layer_of_interest, output_layer_of_interest=output_layer_of_interest,
            non_linearity_method=non_linearity_method, target_output_values=target_output_values,
            target_output_index=target_output_index
        )

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_output_values = target_output_values.cuda() if target_output_values is not None else None

        deltas = 0 if deltas_per_baseline else None
        if not hasattr(self, "reference_activation_lists") or self.reference_activation_lists is None:
            contributions, layer_for_delta_calculation = self.attribute_single_reference(
                input_layer_of_interest, loss_function, non_linearity_method, output_layer_of_interest,
                target_output_index, target_output_values, input_tensor
            )
        else:
            contributions_list = []
            layer_for_delta_calculation = output_layer_of_interest
            for i in range(len(self.reference_activation_lists["input"])):
                # iteration over reference activations for each provided reference input
                self.reference_activation = {layer_name: self.reference_activation_lists[layer_name][i] for layer_name in self.reference_activation_lists.keys()}
                contribution, layer_for_delta_calculation = self.attribute_single_reference(
                    input_layer_of_interest, loss_function, non_linearity_method, output_layer_of_interest,
                    target_output_index, target_output_values, input_tensor
                )
                contributions_list.append(contribution)
                if deltas_per_baseline:
                    deltas += self.get_deltas(contributions=contribution, target_output_index=target_output_index,
                                              output_layer_of_interest=layer_for_delta_calculation)
            length = len(contributions_list)
            contribution_sum = sum(contributions_list)
            contributions = contribution_sum / length
            deltas = deltas / length if deltas_per_baseline else None

        deltas = self.get_deltas(contributions=contributions, output_layer_of_interest=layer_for_delta_calculation,
                                 target_output_index=target_output_index) if deltas is None else deltas
        delattr(self, "forward_activations")
        return contributions, deltas

    def attribute_single_reference(self, input_layer_of_interest, loss_function, non_linearity_method,
                                   output_layer_of_interest, target_output_index, target_output_values, input_tensor):
        self.set_diff_from_ref(input_tensor)
        if target_output_values is not None:
            multipliers, error_layer_name = loss.reverse_loss(dl=self, loss_function=loss_function,
                                                              target_values=target_output_values,
                                                              non_linearity_method=non_linearity_method)
            layer_for_delta_calculation = error_layer_name
        else:
            # initial multipliers need to conform to the output layer of interest shape
            multipliers = self.set_initial_multiplier(output_layer_of_interest=output_layer_of_interest)
            layer_for_delta_calculation = output_layer_of_interest
        first_multiplier_calculation = True
        start_layer_found = False
        for layer_name, layer in reversed(self.model):
            # continue until output layer of interest
            if not start_layer_found and layer_name != output_layer_of_interest:
                continue
            elif layer_name == output_layer_of_interest:
                start_layer_found = True
            multipliers = self.apply_rule(layer=layer, layer_name=layer_name, multipliers=multipliers,
                                          non_linearity_method=non_linearity_method)
            if first_multiplier_calculation and target_output_index is not None:
                multipliers = multipliers[:, target_output_index:target_output_index + 1]
                first_multiplier_calculation = False
            if layer_name == input_layer_of_interest:
                break
        contributions = self.calculate_contributions(input_layer_of_interest=input_layer_of_interest,
                                                     multipliers=multipliers)
        delattr(self, "diff_from_ref")
        return contributions, layer_for_delta_calculation
