from typing import Tuple, List, Dict, Callable
from collections import OrderedDict
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from abc import ABC
import math
from functools import reduce
import operator

sys.path.append(os.path.dirname(__file__))
import parsing as ps


class DeepLiftClass:
    model: ps.LoadedModel
    reference_activation: Dict[str, torch.Tensor] or None

    def __init__(self, model: ps.LoadedModel, reference_value: torch.Tensor) -> None:
        """
        initializes the DeepLiftClass and sets the model and corresponding reference activations

        Args:
            model: model for which explanations should be calculated
            reference_value: baseline value for the input. All explanations will be relative to this value and its
                activations / outputs.
        """
        self.model = model
        self.set_reference_activation_single(reference_value)

    def get_forward_activations(self, x: torch.Tensor) -> Dict:
        """
        does a forward pass of the input and returns the activations of each layer in a single dict

        Args:
            x: the input

        Returns:
            dict with input under input and the activation after each layer under the layer layer_name
        """
        activations = {"input": x}
        for layer_name, layer in self.model:
            # get layer original name from recursive module
            x = layer(x)
            activations[layer_name] = x
        return activations

    def set_reference_activation_full_average(self, reference_dataset: torch.utils.data.Dataset, batch_size: int) \
            -> None:
        """
        creates the reference activation of the model by forward propagating the reference inputs and averaging at each
        layer. Result is saved as reference activation dictionary. !!!NOT TO BE USED!!!

        Args:
            reference_dataset: input to compare the activity of the model against
            batch_size: batch size for the forward pass
        """
        if hasattr(self, "reference_activation") and self.reference_activation is not None:
            raise AttributeError("reference activation already set!")

        loader = torch.utils.data.DataLoader(reference_dataset, batch_size=batch_size, shuffle=False)
        activations: Dict[str, torch.Tensor] = {}
        for inputs, target in loader:
            curr_activations = self.get_forward_activations(inputs)
            # add the activations of the current batch to the activations of the previous batches
            for layer_name, activation in curr_activations.items():
                if layer_name in activations.keys():
                    activations[layer_name] = torch.cat((activations[layer_name], activation))
                else:
                    activations[layer_name] = activation

        # sum up the activations of the batches
        for layer_name, activation in activations.items():
            activations[layer_name] = torch.sum(activation, dim=0)
            # normalize the activations
            activations[layer_name] = activations[layer_name] / len(reference_dataset)

        self.reference_activation = activations

    def set_reference_activation_single(self, reference_input: torch.Tensor) -> None:
        """
        sets the reference activation for a single input value

        Args:
            reference_input: value to set the reference activation for
        """
        if hasattr(self, "reference_activation") and self.reference_activation is not None:
            raise AttributeError("reference activation already set!")
        self.reference_activation = {"input": reference_input}
        activations = self.get_forward_activations(reference_input)
        self.reference_activation.update(activations)

    def get_diff_from_ref(self, input_tens: torch.Tensor) -> Dict:
        """
        calculates the difference between the forward activations of the input and the reference activation

        Args:
            input_tens: input values to calculate the difference for

        Returns:
            tensor with the difference between the forward activations and the reference activation
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet!")
        forward_activations = self.get_forward_activations(input_tens)
        if not forward_activations.keys() == self.reference_activation.keys():
            raise ValueError("keys for forward activation and reference activation dont match!")
        differences = {}
        for (layer_name, reference_val) in self.reference_activation.items():
            differences[layer_name] = forward_activations[layer_name].sub(reference_val)
        return differences

    def get_deltas(self, contributions: torch.Tensor, output_layer_of_interest: str,
                   forward_activations: Dict[str, torch.Tensor], target: int = None) -> torch.Tensor:
        """
        calculates the difference between the summed up explanations and the observed outputs

        Args:
            contributions: contributions of every input for every output in each batch. (shape: batch_size x n(outputs)
            x shape(inputs))
            output_layer_of_interest: layer for which the contributions and deltas are calculated
            forward_activations: activations of the neural net for the forward pass of the input
            target: target class for which the contributions should be calculated as index of the output vector

        Returns:
            Tensor of shape batch_size x n(outputs) containing the difference between the summed up explanations and the
            observed outputs for each output in each batch

        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet!")
        summed_contributions = torch.sum(contributions, dim=tuple(range(2, len(contributions.shape))))
        reference_output = self.reference_activation[output_layer_of_interest]
        if target:
            reference_output = reference_output[:, target]
        predictions = torch.add(summed_contributions, reference_output)
        output = forward_activations[output_layer_of_interest]
        if target:
            output = output[:, target:target+1]
        deltas = torch.sub(predictions, output)
        return deltas

    def linear_rule_linear(self, current_layer: torch.jit._script.RecursiveScriptModule,
                           previous_multipliers: torch.Tensor = None) -> torch.Tensor:
        """
        calculates the multipliers for the current linear layer using the linear rule

        Args:
            current_layer: layer for which the multipliers should be calculated
            previous_multipliers: multipliers of the previous layer

        Returns:
            tensor with the multipliers for the current layer
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet")
        weights_transposed = current_layer.weight
        current_multipliers = torch.matmul(previous_multipliers, weights_transposed)
        return current_multipliers

    @staticmethod
    def set_up_conv_transpose_layer(current_layer: torch.jit._script.RecursiveScriptModule, layer_weights: nn.Parameter) \
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

    def linear_rule_conv(self, current_layer: torch.jit._script.RecursiveScriptModule,
                         previous_multipliers: torch.Tensor = None) -> torch.Tensor:
        """
        calculates the multipliers for the current conv layer using the linear rule

        Args:
            current_layer: layer for which the multipliers should be calculated
            previous_multipliers: multipliers of the previous layer

        Returns:
            tensor with the multipliers for the current layer
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet")
        layer_weights = current_layer.weight

        conv_transpose = self.set_up_conv_transpose_layer(current_layer=current_layer, layer_weights=layer_weights)
        multipliers = []
        for output_dim in range(previous_multipliers.shape[1]):
            current_multipliers = conv_transpose(previous_multipliers[:, output_dim])
            current_multipliers = current_multipliers.unsqueeze(1)
            multipliers.append(current_multipliers)
        result_multipliers = torch.cat(multipliers, dim=1)
        return result_multipliers

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_avg_pos_slices(i, j, k):
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

    @staticmethod
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

    @staticmethod
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

    def reverse_averaging(self, dimensionality: int, input_tensor: torch.Tensor,
                          multipliers: torch.Tensor, kernel_size: List[int],
                          stride: List[int]) -> torch.Tensor:
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
        output = self.repeat_to_match_shapes(for_first_dim=multipliers, to_repeat=output)
        ranges = self.get_ranges(dimensionality=dimensionality, output_shape=multipliers.shape)
        # loops walk over the previous multipliers, while the stride and kernel size are used to calculate the
        # corresponding in the previous layer activation
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    # weights for each position in the kernel are the same: 1 / (kernel_size_total)
                    # the correct position for output and multipliers are selected by the slicing indices
                    slicing_indices = self.get_kernel_pos_slices(i, j, k, kernel_size, stride)
                    kernel_size_total = reduce(operator.mul, kernel_size, 1)
                    avg_slice = self.get_avg_pos_slices(i, j, k)
                    division_result = multipliers[avg_slice] / kernel_size_total
                    division_result = torch.reshape(division_result, division_result.shape + (1, 1))
                    output[slicing_indices] = division_result
        return output

    @staticmethod
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

    def avgpool(self, current_layer: torch.jit._script.RecursiveScriptModule, prev_layer_activation: torch.Tensor,
                previous_multipliers: torch.Tensor = None) -> torch.Tensor:
        """
        calculates the multipliers for the current avgpool layer by calculating the reverse effects of the avgpooling

        Args:
            current_layer: current avgpool layer to calculate the multipliers for
            prev_layer_activation: outputs from the forward pass of the input for the layer one closer to the input
            previous_multipliers: multipliers of the previous layer (one closer to the output)

        Returns:
            multipliers for the current layer
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet")
        # get attributes of current avgpool layer
        dimensionality, kernel_size, padding, stride = self.get_attributes_avg_pool(current_layer)

        # apply padding to the previous layer activation to simulate the padding that was applied in the forward pass
        prev_layer_activation = self.apply_padding(padding=padding, to_pad=prev_layer_activation)

        # calculate the contributions of the previous layer activations to the output of the current layer
        output = self.reverse_averaging(dimensionality=dimensionality, kernel_size=kernel_size, stride=stride,
                                        input_tensor=prev_layer_activation, multipliers=previous_multipliers)

        # remove padding from the output
        output = self.remove_padding(padded=output, padding=padding)
        return output

    @staticmethod
    def get_max_unpool_layers(current_layer: torch.jit._script.RecursiveScriptModule) -> Tuple[
                                                                                             nn.MaxPool1d, nn.MaxUnpool1d] or \
                                                                                         Tuple[
                                                                                             nn.MaxPool2d, nn.MaxUnpool2d] or \
                                                                                         Tuple[
                                                                                             nn.MaxPool3d, nn.MaxUnpool3d]:
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

    def maxpool(self, current_layer: torch.jit._script.RecursiveScriptModule, prev_layer_activation: torch.Tensor,
                previous_multipliers: torch.Tensor = None) -> torch.Tensor:
        """
        calculates the multipliers for the current maxpool layer

        Args:
            current_layer: the current maxpool layer
            prev_layer_activation: activations from the forward pass of the input for the layer one closer to the
                input
            previous_multipliers: multipliers of the previous layer

        Returns:
            multipliers for the current layer
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet")
        forward_layer, reverse_layer = self.get_max_unpool_layers(current_layer)
        # get indices and unpool
        output, indexes = forward_layer(prev_layer_activation)
        # indexes = self.repeat_to_match_shapes(for_first_dim=previous_multipliers, to_repeat=indexes)
        to_concatenate = []
        for output_dim in range(previous_multipliers.shape[1]):
            unpooled = reverse_layer(previous_multipliers[:, output_dim], indexes,
                                     output_size=prev_layer_activation.size())
            unpooled = unpooled.unsqueeze(1)
            to_concatenate.append(unpooled)
        current_multipliers = torch.cat(to_concatenate, dim=1)
        return current_multipliers

    @staticmethod
    def flatten_reverse(shape: List[int], previous_multipliers: torch.Tensor) -> torch.Tensor:
        """
        reverts the flattening of the multipliers

        Args:
            shape: shape of the layer
            previous_multipliers: multipliers for the layer

        Returns:
            unflattened multipliers
        """
        output_dims = previous_multipliers.shape[1]
        shape = list(shape)
        shape.insert(1, output_dims)
        current_multipliers = previous_multipliers.view(shape)
        return current_multipliers

    def rescale_rule(self, current_layer: str, preceding_layer: str, difference_from_reference: Dict[str, torch.Tensor],
                     forward_activations: Dict[str, torch.Tensor], previous_multipliers: torch.Tensor = None) \
            -> torch.Tensor:
        """
        calculates the multipliers for the current layer using the rescale rule

        Args:
            current_layer: layer for which the multipliers should be calculated
            preceding_layer: layer one closer to the input
            difference_from_reference: difference of the activation of the current layer compared to the reference
            forward_activations: activations from the forward pass
            previous_multipliers: multipliers of the previous layer

        Returns:
            multipliers for the current layer
        """
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet")
        delta_x = difference_from_reference[preceding_layer]
        delta_y = difference_from_reference[current_layer]

        # component wise division of delta_y by delta_x
        current_multipliers = torch.div(delta_y, delta_x)

        # use gradient as multipliers if delta_x is too close to zero (to avoid instability)
        mask = torch.abs(delta_x) < 1e-6
        if torch.any(mask):
            gradient = self.calculate_gradient(current_layer=current_layer, preceding_layer=preceding_layer,
                                               forward_activations=forward_activations)
            current_multipliers[mask] = gradient[mask]

        current_multipliers = self.repeat_to_match_shapes(to_repeat=current_multipliers,
                                                          for_first_dim=previous_multipliers)
        # element wise multiplication of the previous multipliers with the current multipliers
        propagated_multipliers = previous_multipliers * current_multipliers
        return propagated_multipliers

    @staticmethod
    def calculate_gradient(current_layer: str, preceding_layer: str, forward_activations: Dict[str, torch.Tensor]) \
            -> torch.Tensor:
        """
        calculates the gradient of the values in the current layer with respect to the values in the preceding layer

        Args:
            current_layer: name of the current layer
            preceding_layer: name of the preceding layer
            forward_activations: activations from the forward pass

        Returns:
            gradient of the current layer with respect to the preceding layer
        """
        preceding_layer_activation = forward_activations[preceding_layer]
        current_layer_activation = forward_activations[current_layer]
        gradient = torch.autograd.grad(current_layer_activation, preceding_layer_activation,
                                       grad_outputs=torch.ones_like(current_layer_activation))
        return gradient[0]

    def check_and_convert_input_attribute(self, input_layer_of_interest: str or int,
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
        if not hasattr(self, "reference_activation"):
            raise AttributeError("reference activation was not set yet!")

        if non_linearity_method == "rescale":
            non_linearity_method = self.rescale_rule
        else:
            raise ValueError("method not supported!")

        return input_layer_of_interest, output_layer_of_interest, non_linearity_method

    def calculate_contributions(self, diff_from_ref: Dict[str, torch.Tensor], input_layer_of_interest: str,
                                multipliers: torch.Tensor) -> torch.Tensor:
        """
        calculates the contributions of the input to the output of the layer of interest.

        Args:
            diff_from_ref: For each layer the difference of reference to the forward activation.
            input_layer_of_interest: name of the input layer to calculate the contributions for
            multipliers: the multipliers for the input layer

        Returns:
            the contributions of the input to the output of the layer of interest
        """
        # multiply the multipliers with the difference from the reference activation to get contributions
        diff_layer = self.model.get_previous_layer_name(input_layer_of_interest)
        differences = diff_from_ref[diff_layer]
        # adjust shape of differences to match multipliers (batch x output features x input shape)
        differences = differences.unsqueeze(1)
        repetitions = [1] * len(differences.shape)
        repetitions[1] = multipliers.shape[1]
        differences = differences.repeat(repetitions)
        contributions = torch.mul(multipliers, differences)
        return contributions

    def attribute(self, input_tensor: torch.Tensor, input_layer_of_interest: str or int = None,
                  output_layer_of_interest: str or int = None, target_output_index: int = None,
                  non_linearity_method: str = "rescale", loss_function: str = "mse",
                  target_output_values: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        Returns: a Tuple (contribs, deltas) containing the contributions of the input layer of interest  to the output
        of the output layer of interest as contribs and the difference between the summed up contributions and the
        observed output as deltas.
        """

        input_layer_of_interest, output_layer_of_interest, non_linearity_method = self.check_and_convert_input_attribute(
            input_layer_of_interest=input_layer_of_interest, output_layer_of_interest=output_layer_of_interest,
            non_linearity_method=non_linearity_method, target_output_values=target_output_values,
            target_output_index=target_output_index
        )

        first_multiplier_calculation = True
        start_layer_found = False
        forward_activations = self.get_forward_activations(input_tensor)
        diff_from_ref = self.get_diff_from_ref(input_tensor)
        if target_output_values is not None:
            multipliers = self.reverse_loss(loss_function=loss_function, target_values=target_output_values,
                                            diff_from_ref=diff_from_ref, forward_activations=forward_activations,
                                            non_linearity_method=non_linearity_method)
        else:
            # initial multipliers need to conform to the output layer of interest shape
            multipliers = self.set_initial_multiplier(forward_activations=forward_activations,
                                                      output_layer_of_interest=output_layer_of_interest
                                                      )

        for layer_name, layer in reversed(self.model):
            # continue until output layer of interest
            if not start_layer_found and layer_name != output_layer_of_interest:
                continue
            elif layer_name == output_layer_of_interest:
                start_layer_found = True
            multipliers = self.apply_rule(diff_from_ref=diff_from_ref, forward_activations=forward_activations,
                                          layer=layer, layer_name=layer_name, multipliers=multipliers,
                                          non_linearity_method=non_linearity_method)

            if first_multiplier_calculation and target_output_index is not None:
                multipliers = multipliers[:, target_output_index:target_output_index + 1]
                first_multiplier_calculation = False

            if layer_name == input_layer_of_interest:
                break

        contributions = self.calculate_contributions(diff_from_ref=diff_from_ref,
                                                     input_layer_of_interest=input_layer_of_interest,
                                                     multipliers=multipliers)
        deltas = self.get_deltas(contributions=contributions, forward_activations=forward_activations,
                                 output_layer_of_interest=output_layer_of_interest, target=target_output_index)
        return contributions, deltas

    def apply_rule(self, diff_from_ref: Dict[str, torch.Tensor], forward_activations: Dict[str, torch.Tensor],
                   layer: torch.jit._script.RecursiveScriptModule, layer_name: str, multipliers: torch.Tensor,
                   non_linearity_method: Callable) -> torch.Tensor:
        """
        applies one rule for calculating the multipliers for the current layer

        Args:
            diff_from_ref: difference from reference for each layer
            forward_activations: activations of each layer
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
            multipliers = self.linear_rule_linear(current_layer=layer, previous_multipliers=multipliers)
        elif layer_type.startswith("Conv"):
            multipliers = self.linear_rule_conv(current_layer=layer, previous_multipliers=multipliers)
        elif layer_type.startswith("MaxPool"):
            output_from_prev_layer = forward_activations[previous_layer_name]
            multipliers = self.maxpool(current_layer=layer, prev_layer_activation=output_from_prev_layer,
                                       previous_multipliers=multipliers)
        elif layer_type.startswith("AvgPool"):
            output_from_prev_layer = forward_activations[previous_layer_name]
            multipliers = self.avgpool(current_layer=layer, prev_layer_activation=output_from_prev_layer,
                                       previous_multipliers=multipliers)
        elif layer_type == "Flatten":
            diff_from_ref_prev_layer = diff_from_ref[previous_layer_name]
            multipliers = self.flatten_reverse(previous_multipliers=multipliers,
                                               shape=diff_from_ref_prev_layer.shape)
        elif layer_type in ["ReLU", "Sigmoid"]:
            multipliers = non_linearity_method(current_layer=layer_name, preceding_layer=previous_layer_name,
                                               previous_multipliers=multipliers,
                                               difference_from_reference=diff_from_ref,
                                               forward_activations=forward_activations)
        else:
            raise ValueError(f"layer type \"{layer_type}\" not supported!")
        return multipliers

    def set_initial_multiplier(self, forward_activations: Dict[str, torch.Tensor], output_layer_of_interest: str) -> \
            torch.Tensor:
        """
        sets the initial multipliers for the output layer of interest, adjusted for the correct shape of batchsize x
        output features x output features

        Multipliers are given the shape batchsize x output features x output features. This means that after back
        propagation, the multipliers for the first layer are already in the correct shape (batchsize x output features x
        input features). This for now only works for layers with one dimensional output.

        Args:
            forward_activations: the forward activations of the network
            output_layer_of_interest: output layer to calculate the contributions for. Initial multipliers are need to
                be adjusted to the shape of the out of this layer. Layer must have one dimensional output.

        Returns:
            the initial multipliers
        """
        shape = forward_activations[output_layer_of_interest].shape
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

    def reverse_loss(self, loss_function: str, target_values: torch.Tensor, diff_from_ref: Dict[str, torch.Tensor],
                     forward_activations: Dict[str, torch.Tensor], non_linearity_method: Callable) -> torch.Tensor:
        """
        calculates multipliers for the given loss function

        Args:
            loss_function: type of loss used
            target_values: target values for the output of the network. Needed for loss calculation.
            diff_from_ref: difference from the reference activation for each layer
            forward_activations: activations for each layer
            non_linearity_method: method to deal with non linearities

        Returns:
            calculated multipliers after the loss function
        """
        if loss_function.lower() in ["mean_squared_error", "mse"]:
            multipliers = self.mse_loss(diff_from_ref=diff_from_ref, forward_activations=forward_activations,
                                        target_output_values=target_values, non_linearity_method=non_linearity_method)
        else:
            raise ValueError(f"loss \"{loss_function}\" not supported!")
        return multipliers

    def mse_loss(self, diff_from_ref: Dict[str, torch.Tensor], forward_activations: Dict[str, torch.Tensor],
                 target_output_values: torch.Tensor, non_linearity_method: Callable) -> torch.Tensor:
        """
        calculates the multipliers for the mean squared error loss

        Args:
            diff_from_ref: difference from the reference activation for each layer
            forward_activations: activations for each layer
            target_output_values: target values for the output of the network. Needed for loss calculation.
            non_linearity_method: method to deal with non linearities

        Returns:
            calculated multipliers after the loss function
        """
        output_layer_name, _ = self.model[-1]
        first_layer_name, _ = self.model[0]
        reference_targets = target_output_values

        # calculate reference loss
        output_layer_activation_reference = self.reference_activation[output_layer_name]
        output_layer_deviance_reference = output_layer_activation_reference - reference_targets
        squared_reference = torch.pow(output_layer_deviance_reference, 2)
        self.reference_activation["squared"] = squared_reference
        self.reference_activation["mse"] = torch.mean(squared_reference)

        # calculate loss for given input
        output_layer_activation = forward_activations[output_layer_name]
        output_layer_deviance = output_layer_activation - target_output_values
        squared_output = torch.pow(output_layer_deviance, 2)
        mse_loss = torch.mean(squared_output)

        forward_activations["squared"] = squared_output
        forward_activations["mse"] = mse_loss
        diff_from_ref["squared"] = squared_output - squared_reference
        diff_from_ref["mse"] = mse_loss - self.reference_activation["mse"]

        # multipliers from averaging over the squares
        shape = list(output_layer_activation.shape)
        shape[0] = 1
        multipliers_for_averaging_layer = torch.full(size=shape, fill_value=1. / output_layer_activation.numel())
        multipliers_for_squared_layer = non_linearity_method(current_layer="squared", preceding_layer=output_layer_name,
                                                             previous_multipliers=multipliers_for_averaging_layer,
                                                             difference_from_reference=diff_from_ref,
                                                             forward_activations=forward_activations)

        return multipliers_for_squared_layer


if __name__ == "__main__":
    test_model = nn.Sequential(
        nn.Linear(3, 1, bias=False)
    )
    test_model[0].weight.data = torch.tensor([[1., 1., 1.]], dtype=torch.float)
    test_torch_script = torch.jit.script(test_model)
    test_model = ps.SequentialLoadedModel(test_torch_script)
    dl = DeepLiftClass(model=test_model, reference_value=torch.tensor([[0., 0., 0.]], dtype=torch.float))
    attributions = dl.attribute(input_tensor=torch.tensor([[1., 1., 1.], [2, 2, 2]]))
    print(f"contributions: {attributions[0]}")
    print(f"deltas: {attributions[1]}")
