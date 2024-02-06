import numpy as np
import unittest
import warnings
from typing import List
import torch
from torch import nn
import pickle
import __main__
import os
import shap

import tests.comparison as comparison
import tests.regression_models as regression_models
import tests.classifier_models as classifier_models
import tests.models as models
from tests.regression_models import MinDataSetFCNN
from src.DeepLIFT_Lamarr.deeplift import DeepLiftClass
from src.DeepLIFT_Lamarr.parsing import SequentialLoadedModel


class IntegrationTests(unittest.TestCase):

    def test_non_convolution_nets(self):
        __main__.MinDataSetFCNN = MinDataSetFCNN
        further_tests = [
            ("super_simple_sequential_linear_multiple_outputs", 3),
            ("super_simple_sequential_non_linear_multiple_outputs", 3)
        ]
        results = comparison.non_convolution_comparisons(additional_tests=further_tests, efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear_1d(self):
        results = comparison.convolution_comparisons([(1, 3, "super_simple_convolutional_non_linear_1d")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear(self):
        results = comparison.convolution_comparisons([(2, 3, "super_simple_convolutional_non_linear")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear_3d(self):
        results = comparison.convolution_comparisons([(3, 3, "super_simple_convolutional_non_linear_3d")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_simple_mixed_net_multiple_outputs(self):
        results = comparison.convolution_comparisons([(2, 3, "simple_mixed_net_multiple_outputs")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_linear(self):
        results = comparison.convolution_comparisons([(2, 3, "super_simple_convolutional_linear")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_1d_max_pooling(self):
        results = comparison.convolution_comparisons([(1, 5, "super_simple_convolutional_1d_max_pooling")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_2d_max_pooling(self):
        results = comparison.convolution_comparisons([(2, 5, "super_simple_convolutional_2d_max_pooling")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_3d_max_pooling(self):
        results = comparison.convolution_comparisons([(3, 5, "super_simple_convolutional_3d_max_pooling")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_2d_max_pooling_multi_channel(self):
        results = comparison.convolution_comparisons(
            [(2, 5, "super_simple_convolutional_2d_max_pooling_multi_channel")], efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_2d_avg_pooling_multi_channel(self):
        results = comparison.convolution_comparisons(
            [(2, 5, "super_simple_convolutional_2d_avg_pooling_multi_channel")], efficient_target_calc=False)
        self.check_results(results)

    def test_super_simple_convolutional_2d_avg_pooling(self):
        results = comparison.convolution_comparisons([(2, 5, "super_simple_convolutional_2d_avg_pooling")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_super_short_2d_avg_pooling(self):
        results = comparison.convolution_comparisons([(2, 4, "super_short_2d_avg_pooling")],
                                                     efficient_target_calc=False)
        self.check_results(results)

    def test_sigmoid(self):
        results = comparison.test_sigmoid()
        self.check_results(results)

    def test_MNIST(self):
        results = comparison.compare_MNIST(efficient_target_calc=False)
        self.check_results(results)

    def test_non_convolution_nets_eff(self):
        __main__.MinDataSetFCNN = MinDataSetFCNN
        # print(__name__)
        further_tests = [
            ("super_simple_sequential_linear_multiple_outputs", 3),
            ("super_simple_sequential_non_linear_multiple_outputs", 3)
        ]
        results = comparison.non_convolution_comparisons(additional_tests=further_tests, efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear_1d_eff(self):
        results = comparison.convolution_comparisons([(1, 3, "super_simple_convolutional_non_linear_1d")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear_eff(self):
        results = comparison.convolution_comparisons([(2, 3, "super_simple_convolutional_non_linear")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_non_linear_3_eff(self):
        results = comparison.convolution_comparisons([(3, 3, "super_simple_convolutional_non_linear_3d")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_simple_mixed_net_multiple_output_eff(self):
        results = comparison.convolution_comparisons([(2, 3, "simple_mixed_net_multiple_outputs")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_linear_eff(self):
        results = comparison.convolution_comparisons([(2, 3, "super_simple_convolutional_linear")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_1d_max_pooling_eff(self):
        results = comparison.convolution_comparisons([(1, 5, "super_simple_convolutional_1d_max_pooling")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_2d_max_pooling_eff(self):
        results = comparison.convolution_comparisons([(2, 5, "super_simple_convolutional_2d_max_pooling")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_3d_max_pooling_eff(self):
        results = comparison.convolution_comparisons([(3, 5, "super_simple_convolutional_3d_max_pooling")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_2d_max_pooling_multi_channel_eff(self):
        results = comparison.convolution_comparisons(
            [(2, 5, "super_simple_convolutional_2d_max_pooling_multi_channel")], efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_2d_avg_pooling_multi_channel_eff(self):
        results = comparison.convolution_comparisons(
            [(2, 5, "super_simple_convolutional_2d_avg_pooling_multi_channel")], efficient_target_calc=True)
        self.check_results(results)

    def test_super_simple_convolutional_2d_avg_pooling_eff(self):
        results = comparison.convolution_comparisons([(2, 5, "super_simple_convolutional_2d_avg_pooling")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_super_short_2d_avg_pooling_eff(self):
        results = comparison.convolution_comparisons([(2, 4, "super_short_2d_avg_pooling")],
                                                     efficient_target_calc=True)
        self.check_results(results)

    def test_MNIST_eff(self):
        results = comparison.compare_MNIST(efficient_target_calc=True)
        self.check_results(results)

    def check_results(self, results: List[comparison.ComparisonResult]):
        for result in results:
            attribution_difference = result.captum_attributions - result.new_dl_attributions
            delta_difference = result.captum_deltas - result.new_dl_deltas
            delta_sum_new = torch.sum(result.new_dl_deltas)
            delta_sum_captum = torch.sum(result.captum_deltas)
            self.assertAlmostEqual(delta_sum_new, delta_sum_captum, delta=1e-5,
                                   msg=f"Test {result.name} failed with a delta sum difference of "
                                       f"{delta_sum_new - delta_sum_captum}")
            self.assertTrue((torch.max(attribution_difference)) < 1e-5, f"Test {result.name} failed with a maximum "
                                                                        f"attribution difference of "
                                                                        f"{torch.max(attribution_difference)}")
            self.assertTrue((torch.max(delta_difference)) < 1e-5, f"Test {result.name} failed with a maximum delta"
                                                                  f"difference of {torch.max(delta_difference)}")

    def test_special_case_rescale(self):
        explainer = models.get_explainer(model_name="super_simple_sequential_non_linear", baseline=torch.zeros(1, 3))
        explanations, deltas = explainer.attribute(input_tensor=torch.tensor([[1., 0., -1.]]))
        # print("explanations:", explanations)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(explanations, torch.tensor([[[0., 0., -0.]]])))

    def test_input_semantics(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        dl_explainer = models.get_explainer("super_simple_sequential_linear", baseline)
        explanations_0 = dl_explainer.attribute(input_tensor, input_layer_of_interest=0)[0]
        explanations_input = dl_explainer.attribute(input_tensor, input_layer_of_interest="input")[0]
        explanations_default = dl_explainer.attribute(input_tensor)[0]
        max_diff_0_default = torch.max(torch.abs(explanations_0 - explanations_default))
        max_diff_input_default = torch.max(torch.abs(explanations_input - explanations_default))
        self.assertAlmostEqual(max_diff_0_default, 0)
        self.assertAlmostEqual(max_diff_input_default, 0)

    def test_output_semantics(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        dl_explainer = models.get_explainer("super_simple_sequential_linear", baseline)
        explanations_output = dl_explainer.attribute(input_tensor, output_layer_of_interest=-1)[0]
        explanations_default = dl_explainer.attribute(input_tensor)[0]
        max_diff = torch.max(torch.abs(explanations_output - explanations_default))
        self.assertAlmostEqual(max_diff, 0)

    def test_input_super_simple_sequential(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_sequential_linear", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, input_layer_of_interest=1)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[6, -3]]], dtype=torch.float32), explanations))

    def test_output_super_simple_sequential(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_sequential_linear", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, output_layer_of_interest=-2)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[2, 2, 2], [1, 1, 1]]], dtype=torch.float32), explanations))

    def test_input_output_super_simple_sequential(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_sequential_non_linear", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, input_layer_of_interest=1,
                                                      output_layer_of_interest=-2)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[6, -3]]], dtype=torch.float32), explanations))

    def test_input_super_simple_sequential_multiple_outputs(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_sequential_non_linear_multiple_outputs", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, input_layer_of_interest=1)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[6, -3], [0, 0]]], dtype=torch.float32), explanations))

    def test_output_super_simple_sequential_multiple_outputs(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_sequential_non_linear_multiple_outputs", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, output_layer_of_interest=-2)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[1, 1, 1], [-1, -1, -1]]], dtype=torch.float32), explanations))

    def test_input_mixed(self):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, input_layer_of_interest=1)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        target = torch.tensor([[[[[1, 0], [0, 0]]]]], dtype=torch.float32)
        self.assertTrue(torch.sum(torch.abs(explanations - target)) < 1e-5)

    def test_output_mixed(self):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor, output_layer_of_interest=-2)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]], dtype=torch.float32),
                                    explanations))

    def test_loss_explanation(self, non_linearity: str = "rescale", print_deltas=False):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net", baseline=baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method=non_linearity,
                                                      target_output_values=torch.tensor([[0]]))
        if print_deltas:
            print(deltas)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        target = torch.tensor([[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]], dtype=torch.float32)
        self.assertTrue(torch.sum(torch.abs(explanations - target)) < 1e-5)

    def test_loss_explanation_2(self, non_linearity: str = "rescale", print_deltas=False):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 1, 1], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method=non_linearity,
                                                      target_output_values=torch.tensor([[0]]))
        if print_deltas:
            print(deltas)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(torch.tensor([[[[2, 2, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32),
                                    explanations[0]))

    def test_loss_explanation_3(self, non_linearity: str = "rescale", print_deltas=False):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method=non_linearity,
                                                      target_output_values=torch.tensor([[2]]))
        if print_deltas:
            print(deltas)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        target = torch.tensor([[[[[-3, 0, 0], [0, 0, 0], [0, 0, 0]]]]], dtype=torch.float32)
        self.assertTrue(torch.sum(torch.abs(explanations - target)) < 1e-5)

    def test_loss_explanation_multi_output(self, non_linearity: str = "rescale", print_deltas=False):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_mixed_net_multiple_outputs", baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method=non_linearity,
                                                      target_output_values=torch.tensor([[0, 0]]))
        target_result = torch.tensor([[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]],
                                     dtype=torch.float32)
        if print_deltas:
            print(deltas)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.sum(torch.abs(explanations - target_result)) < 1e-5)

    def test_linear_reveal_cancel(self):
        baseline = torch.tensor([[0, 0, 0]], dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("linear_relu_min_example", baseline=baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method="reveal-cancel")
        expected_result = torch.tensor([[[0.7500, -0.5000,  0.7500], [-0.5000,  0.5000,  0.0000]]],
                                       dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(expected_result, explanations))

    def test_conv_reveal_cancel(self):
        baseline = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_convolutional_non_linear", baseline=baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method="reveal-cancel")
        expected_result = torch.tensor([[[[[1, 0, 0], [0,  0,  0], [0,  0,  0]]]]],
                                       dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(expected_result, explanations))

    def test_conv_1d_reveal_cancel(self):
        baseline = torch.tensor([[[0, 0, 0]]], dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[1, 1, 1]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("super_simple_convolutional_non_linear_1d", baseline=baseline)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method="reveal-cancel")
        expected_result = torch.tensor([[[[1, 0, 0]]]],
                                       dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(expected_result, explanations))

    def test_flatten_reveal_cancel(self):
        baseline = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer("simple_flatten_before_non_linear", baseline=baseline)
        input_tensor = torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32, requires_grad=True)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method="reveal-cancel")
        expected_result = torch.tensor([[[[[1, 0, 0], [0,  0,  0], [0,  0,  0]]], [[[0, 1, 0], [0,  0,  0], [0,  0,  0]]],
                                       [[[0, 0, 0], [1,  0,  0], [0,  0,  0]]], [[[0, 0, 0], [0,  1,  0], [0,  0,  0]]]]],
                                       dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(expected_result, explanations))

    def test_flatten_first_reveal_cancel(self):
        baseline = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
        dl_explainer = models.get_explainer(model_name="tiny_flatten_first", baseline=baseline)
        input_tensor = torch.ones((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
        explanations, deltas = dl_explainer.attribute(input_tensor=input_tensor, non_linearity_method="reveal-cancel")
        expected_result = torch.tensor([[[[[[0, 1]]]]]], dtype=torch.float32, requires_grad=True)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(expected_result, explanations))

    def test_avg_pool_pos_neg(self):
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_avgpool", baseline=baseline)
        explanations, deltas = explainer.attribute(torch.tensor([[[[0., -1.], [-2., 3.]]]], dtype=torch.float32,
                                                                requires_grad=True),
                                                   input_layer_of_interest=0, non_linearity_method="reveal_cancel")
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        self.assertTrue(torch.equal(explanations, torch.tensor([[[[[0, -.125], [-.25, .375]]]]])))

    def test_max_pool_pos_neg(self):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_maxpool", baseline=baseline)
        explanations, deltas = explainer.attribute(torch.tensor([[[[0., -1., 2], [-2., 3., -4], [0, 0, 0]]]],
                                                                dtype=torch.float32,
                                                                requires_grad=True),
                                                   non_linearity_method="reveal_cancel")
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        # print(deltas)
        self.assertTrue(torch.equal(explanations, torch.tensor([[[[[0, 0, 0], [0, 3, 0], [0, 0, 0]]]]],
                                                               dtype=torch.float32)))

    def test_reveal_cancel_pos_neg(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="rescale_pos_neg_model", baseline=baseline)
        input_values = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        explanations, deltas = explainer.attribute(input_values, non_linearity_method="reveal_cancel")
        target = torch.tensor([[[1, 1, 1], [0, 0, 0]]], dtype=torch.float32)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
        # print(deltas)
        self.assertTrue(torch.equal(explanations, target))

    def test_mse_pos_neg(self):
        self.test_loss_explanation(non_linearity="reveal-cancel", print_deltas=False)
        self.test_loss_explanation_2(non_linearity="reveal-cancel", print_deltas=False)
        self.test_loss_explanation_2(non_linearity="reveal-cancel", print_deltas=False)
        self.test_loss_explanation_multi_output(non_linearity="reveal-cancel", print_deltas=False)

    def test_contribution_calculation_pos_neg(self):
        baseline = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_ReLU_first", baseline=baseline)
        input_values = torch.tensor([[1, 1]], dtype=torch.float32, requires_grad=True)
        explanations, deltas = explainer.attribute(input_values, non_linearity_method="reveal_cancel")
        target = torch.tensor([[[0, 1]]], dtype=torch.float32)
        if torch.cuda.is_available():
            explanations = explanations.cpu()
            deltas = deltas.cpu()
        self.assertTrue(torch.equal(deltas, torch.zeros((1, 1), dtype=torch.float32)))
        self.assertTrue(torch.equal(explanations, target))

    # def test_compare_MNIST_reveal_cancel_mixed(self):
    #     test_loader, train_loader = regression_models.get_MNIST_dataloaders()
    #     baseline = comparison.get_MNIST_avgs(test_loader)
    #     new_dl = models.get_explainer("MNIST_net_mixed", baseline=baseline)
    #     images, _ = next(iter(train_loader))
    #     image = images[0:4]
    #     result, deltas = new_dl.attribute(image, non_linearity_method="reveal_cancel")
    #     delta_sum = torch.sum(torch.abs(deltas))
    #     # print(delta_sum)
    #     self.assertTrue(torch.sum(delta_sum) < .1)

    def test_compare_MNIST_reveal_cancel_avg_pool(self):
        test_loader, train_loader = regression_models.get_MNIST_dataloaders()
        baseline = comparison.get_MNIST_avgs(test_loader)
        new_dl = models.get_explainer("MNIST_net_no_max", baseline=baseline)
        images, _ = train_loader.__iter__().__next__()
        image = images[0:4]
        result, deltas = new_dl.attribute(image, non_linearity_method="reveal_cancel")
        delta_sum = torch.sum(torch.abs(deltas))
        # print(delta_sum)
        self.assertTrue(delta_sum < .1)

    def test_compare_MNIST_rescale_avg_pool(self):
        test_loader, train_loader = regression_models.get_MNIST_dataloaders()
        baseline = comparison.get_MNIST_avgs(test_loader)
        new_dl = models.get_explainer("MNIST_net_no_max", baseline=baseline)
        images, _ = train_loader.__iter__().__next__()
        image = images[0:4]
        result, deltas = new_dl.attribute(image, non_linearity_method="rescale")
        delta_sum = torch.sum(torch.abs(deltas))
        # print(delta_sum)
        self.assertTrue(delta_sum < .1)

    def investigate_delta_progression(self, explainer, input_vals: torch.Tensor, non_linearity_method: str,
                                      print_stuff: bool = False, verbose: bool = False):
        results = []
        for i in range(len(explainer.model)):
            result, deltas = explainer.attribute(input_tensor=input_vals, non_linearity_method=non_linearity_method,
                                                 input_layer_of_interest=i)
            delta_sum = torch.sum(torch.abs(deltas))
            results.append((i, delta_sum, result))
        if print_stuff:
            for i, delta_sum, result in results:
                if verbose:
                    print(f"Layer {i}:")
                    print(F"    sum of deltas: {delta_sum}")
                    print(f"    contributions: {result}")
                else:
                    print(f"Layer {i}: {delta_sum}")
        for _, delta_sum, _ in results:
            self.assertTrue(delta_sum < 1e-3)

    # def test_inspect_deltas_MNIST_reveal_cancel(self):
    #     test_loader, train_loader = regression_models.get_MNIST_dataloaders()
    #     baseline = comparison.get_MNIST_avgs(test_loader)
    #     new_dl = models.get_explainer("MNIST_net_mixed", baseline=baseline)
    #     images, _ = train_loader.__iter__().__next__()
    #     image = images[0:4]
    #     print("reveal_cancel")
    #     self.investigate_delta_progression(explainer=new_dl, input_vals=image, non_linearity_method="reveal_cancel",
    #                                        print_stuff=True)
    #     print("rescale")
    #     self.investigate_delta_progression(explainer=new_dl, input_vals=image, non_linearity_method="rescale",
    #                                        print_stuff=True)
    #     self.assertTrue(True)

    def test_simple_full_net(self):
        models.get_model("sequential_non_linear")
        with open(os.path.join("saved_networks", "sequential_non_linear_training_set.pkl"), 'rb') as f:
            training_set = pickle.load(f)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=False)
        baseline = training_set.average_input
        explainer = models.get_explainer("sequential_non_linear", baseline=baseline)
        images, targets = next(iter(training_loader))
        self.investigate_delta_progression(explainer=explainer, input_vals=images, non_linearity_method="reveal_cancel")

    def test_simplest_classifier(self):
        training_in, training_targets = classifier_models.get_test_classification_dataset_linear()
        training_in = torch.tensor(training_in, dtype=torch.float32)
        baseline = torch.mean(training_in, dim=0, keepdim=True)
        explainer = models.get_explainer(model_name="simplest_classifier", baseline=baseline)
        to_explain = training_in[0:4]
        self.investigate_delta_progression(explainer=explainer, input_vals=to_explain,
                                           non_linearity_method="reveal_cancel")

    def test_simplest_classifier_relu(self):
        training_in, training_targets = classifier_models.get_test_classification_dataset_linear()
        training_in = torch.tensor(training_in, dtype=torch.float32)
        baseline = torch.mean(training_in, dim=0, keepdim=True)
        explainer = models.get_explainer(model_name="simplest_classifier_relu", baseline=baseline)
        to_explain = training_in[0:4]
        self.investigate_delta_progression(explainer=explainer, input_vals=to_explain,
                                           non_linearity_method="reveal_cancel")

    def test_hidden_layer_classifier_relu(self):
        training_in, training_targets = classifier_models.get_test_classification_dataset_linear()
        training_in = torch.tensor(training_in, dtype=torch.float32)
        baseline = torch.mean(training_in, dim=0, keepdim=True)
        explainer = models.get_explainer(model_name="hidden_layer_classifier", baseline=baseline)
        to_explain = training_in[0:4]
        self.investigate_delta_progression(explainer=explainer, input_vals=to_explain,
                                           non_linearity_method="reveal_cancel")

    def test_small_conv_classifier_relu(self):
        baseline = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        explainer = models.get_explainer(model_name="small_conv_classifier_easy_weights", baseline=baseline)
        to_explain = torch.tensor([[[[0.6, -0.2, -0.5], [0.1, -0.1, .3], [2, -1.5, -1]]]], dtype=torch.float32)
        explainer.reference_activation = {layer_name: reference_list[0] for layer_name, reference_list
                                          in explainer.reference_activation_lists.items()}
        explainer.set_diff_from_ref(to_explain)
        delattr(explainer, "diff_from_ref")
        delattr(explainer, "forward_activations")
        self.investigate_delta_progression(explainer=explainer, input_vals=to_explain,
                                           non_linearity_method="reveal_cancel", verbose=True)

    def compare_dropout(self, non_linearity_method: str, print_stuff: bool = False):
        baseline = torch.zeros((1, 3), dtype=torch.float32)
        no_dropout_model = models.get_model(model_name="sequential_non_linear_random_weights")
        dropout_model = models.get_model(model_name="sequential_non_linear_random_weights_dropout")
        no_dropout_model[0].weight = dropout_model[0].weight
        no_dropout_model[0].bias = dropout_model[0].bias
        no_dropout_model[2].weight = dropout_model[3].weight
        no_dropout_model[2].bias = dropout_model[3].bias

        no_dropout_model_ts = torch.jit.script(no_dropout_model)
        dropout_model_ts = torch.jit.script(dropout_model)

        no_dropout_explainer = DeepLiftClass(SequentialLoadedModel(no_dropout_model_ts), reference_value=baseline)
        dropout_explainer = DeepLiftClass(SequentialLoadedModel(dropout_model_ts), reference_value=baseline)
        input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        output_no_dropout = no_dropout_model(input_tensor)
        output_dropout = dropout_model(input_tensor)
        dropout_explanations, dropout_deltas = dropout_explainer.attribute(input_tensor, non_linearity_method=non_linearity_method)
        no_dropout_explanations, no_dropout_deltas = no_dropout_explainer.attribute(input_tensor, non_linearity_method=non_linearity_method)
        if print_stuff:
            print(f"no dropout output: {output_no_dropout}")
            print(f"dropout output: {output_dropout}")
            print(f"dropout explanations: {dropout_explanations}")
            print(f"no dropout explanations: {no_dropout_explanations}")
            print(f"dropout deltas: {dropout_deltas}")
            print(f"no dropout deltas: {no_dropout_deltas}")
        self.assertTrue(torch.equal(output_no_dropout, output_dropout))
        self.assertTrue(torch.equal(no_dropout_explanations, dropout_explanations))
        self.assertTrue(torch.equal(no_dropout_deltas, dropout_deltas))
        self.assertTrue(torch.sum(torch.abs(dropout_deltas)) < 1e-5)

    def test_compare_dropout(self):
        self.compare_dropout(non_linearity_method="reveal_cancel")
        self.compare_dropout(non_linearity_method="rescale")

    @staticmethod
    def run_non_zero_padding():
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        explainer = models.get_explainer(model_name="non_zero_padding", baseline=baseline, shap=False)
        input_tensor = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32)
        explanations, deltas = explainer.attribute(input_tensor, non_linearity_method="reveal_cancel")

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.run_non_zero_padding()
            self.assertEqual(1, len(w))
            self.assertTrue("padding" in str(w[0].message))

    def test_compare_MNIST_reveal_cancel_avg_pool_shap(self):
        test_loader, train_loader = regression_models.get_MNIST_dataloaders()
        baseline = comparison.get_MNIST_avgs(test_loader)
        new_dl = models.get_explainer("MNIST_net_no_max", baseline=baseline, shap=False)
        images, _ = train_loader.__iter__().__next__()
        image = images[0:4]
        result, deltas = new_dl.attribute(image, non_linearity_method="reveal_cancel")
        delta_sum = torch.sum(torch.abs(deltas))
        # print(delta_sum)
        self.assertTrue(delta_sum < .1)

    def test_compare_MNIST_rescale_avg_pool_shap(self):
        test_loader, train_loader = regression_models.get_MNIST_dataloaders()
        baseline = comparison.get_MNIST_avgs(test_loader)
        new_dl = models.get_explainer("MNIST_net_no_max", baseline=baseline)
        images, _ = train_loader.__iter__().__next__()
        image = images[0:4]
        result, deltas = new_dl.attribute(image, non_linearity_method="rescale")
        delta_sum = torch.sum(torch.abs(deltas))
        # print(delta_sum)
        self.assertTrue(delta_sum < .1)

    def compare_to_shap(self, model_object: nn.Module, torch_script_model: torch.jit._script.RecursiveScriptModule,
                        baselines: torch.Tensor, input_tensor: torch.Tensor):
        explainer_new = DeepLiftClass(model=torch_script_model, reference_value=baselines, shap=True)
        explainer_shap = shap.DeepExplainer(model=model_object, data=baselines)
        for i in range(input_tensor.shape[0]):
            explanation_new, deltas_standard = explainer_new.attribute(input_tensor=input_tensor[i:i + 1])
            _, deltas_per_baseline = explainer_new.attribute(input_tensor=input_tensor[i:i + 1], deltas_per_baseline=True)
            explanation_shap = explainer_shap.shap_values(input_tensor[i:i + 1])
            shap_tens = torch.tensor(np.concatenate(explanation_shap, axis=0), dtype=torch.float32).unsqueeze(0)
            self.assertTrue(torch.allclose(explanation_new, shap_tens, atol=1e-5, rtol=0))
            self.assertTrue(torch.allclose(deltas_per_baseline, torch.zeros_like(deltas_per_baseline), atol=1e-5, rtol=0))

    def test_basic_shap_comparison(self):
        model_object = models.get_model("linear_relu_min_example")
        torch_script_model = models.get_model("linear_relu_min_example", return_torch_script=True)
        baselines = torch.cat([torch.zeros((1, 3), dtype=torch.float32), torch.ones((1, 3), dtype=torch.float32)])
        input_tensor = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float32)
        self.compare_to_shap(model_object=model_object, torch_script_model=torch_script_model, baselines=baselines,
                             input_tensor=input_tensor)

    def test_MNIST_shap_comparison(self):
        model_object = models.get_model("MNIST_net_no_max")
        torch_script_model = models.get_model("MNIST_net_no_max", return_torch_script=True)
        test_loader, train_loader = regression_models.get_MNIST_dataloaders()
        images, _ = train_loader.__iter__().__next__()
        baselines = images[0:50]
        input_tensor = images[50:55]
        self.compare_to_shap(model_object=model_object, torch_script_model=torch_script_model, baselines=baselines,
                             input_tensor=input_tensor)


if __name__ == "__main__":
    unittest.main()
