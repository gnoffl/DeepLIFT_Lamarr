import unittest
from typing import Tuple
import torch
import tests.models as models
import src.deeplift.non_linear as non_linear
import src.deeplift.flatten as flatten
import src.deeplift.linear as linear
import src.deeplift.avgpool as avgpool


class DeepliftTests(unittest.TestCase):

    def test_get_pos_neg_contributions_linear(self):
        explainer = models.get_explainer(model_name="linear_relu_min_example", baseline=torch.zeros(1, 3), shap=False)
        explainer.set_diff_from_ref(torch.tensor([[1., 1., 1.]]))
        pos, neg = non_linear.get_pos_neg_contributions_linear(dl=explainer, layer_name=explainer.model[0][0])
        if torch.cuda.is_available():
            pos = pos.cpu()
            neg = neg.cpu()
        self.assertTrue(torch.equal(pos, torch.tensor([[2., 1.]])))
        self.assertTrue(torch.equal(neg, torch.tensor([[-1., -1.]])))

    def test_calculate_reveal_cancel_contributions(self):
        explainer = models.get_explainer(model_name="linear_relu_min_example",
                                                    baseline=torch.tensor([[-1., -1., -1.]]), shap=False)
        explainer.set_diff_from_ref(torch.tensor([[1., 1., 1.]]))
        pos, neg = non_linear.get_pos_neg_contributions_linear(dl=explainer, layer_name=explainer.model[0][0])
        delta_y_plus = non_linear.calculate_reveal_cancel_contributions(dl=explainer,
                                                                        current_layer_name=explainer.model[1][0],
                                                                        main_input_delta=pos, secondary_input_delta=neg)
        delta_y_minus = non_linear.calculate_reveal_cancel_contributions(dl=explainer,
                                                                         current_layer_name=explainer.model[1][0],
                                                                         main_input_delta=neg,
                                                                         secondary_input_delta=pos)
        delta_y_plus = delta_y_plus.cpu()
        delta_y_minus = delta_y_minus.cpu()
        self.assertTrue(torch.equal(delta_y_plus, torch.tensor([[2., 1.]])))
        self.assertTrue(torch.equal(delta_y_minus, torch.tensor([[-1., -1.]])))

    def test_reveal_cancel_rule(self):
        explainer = models.get_explainer(model_name="linear_relu_min_example",
                                                    baseline=torch.tensor([[-1., -1., -1.]]), shap=False)
        explainer.set_diff_from_ref(input_tens=torch.tensor([[1., 1., 1.]]))
        initial_multipliers = explainer.set_initial_multiplier(explainer.model[-1][0])
        multiplier_plus, multiplier_minus = non_linear.reveal_cancel_rule(current_layer_name=explainer.model[-1][0],
                                                                          previous_multipliers=initial_multipliers,
                                                                          dl=explainer)
        # pos_contrib, neg_contrib = explainer.get_pos_neg_contributions_linear(layer_name=explainer.model[0][0])
        # print("pos contrib:", pos_contrib)
        # print("neg contrib:", neg_contrib)
        # print("plus:", multiplier_plus)
        # print("minus:", multiplier_minus)
        #
        # multipliers_neutral = explainer.rescale_rule(current_layer=explainer.model[-1][0],
        #                                              previous_multipliers=initial_multipliers)
        # print("mutip neutral:", multipliers_neutral)
        # print("diff_from_ref_output:", explainer.diff_from_ref[explainer.model[-1][0]])

        multiplier_plus = multiplier_plus.cpu()
        multiplier_minus = multiplier_minus.cpu()
        self.assertTrue(torch.equal(multiplier_plus, torch.tensor([[[.5, 0], [0, .5]]])))
        self.assertTrue(torch.equal(multiplier_minus, torch.tensor([[[.5, 0], [0, .5]]])))

    def test_linear_rule_pos_neg(self, previous_multipliers: Tuple[torch.Tensor, torch.Tensor] = None,
                                 target_result: torch.Tensor = None):
        explainer = models.get_explainer(model_name="linear_relu_min_example",
                                                    baseline=torch.tensor([[-1., -1., -1.]]), shap=False)
        explainer.set_diff_from_ref(input_tens=torch.tensor([[1., 1., 1.]]))
        if previous_multipliers is None:
            previous_multipliers = (torch.tensor([[[.5, 0], [0, .5]]]), torch.tensor([[[.5, 0], [0, .5]]]))
        result = linear.linear_rule_pos_neg(dl=explainer, current_layer=explainer.model[0][1],
                                            current_layer_name=explainer.model[0][0],
                                            previous_multipliers=previous_multipliers)
        if target_result is None:
            target_result = torch.tensor([[[0.5, -0.5, 0.5], [-0.5, 0.5, 0]]], dtype=torch.float32,
                                         requires_grad=True)
        result = result.cpu()
        self.assertTrue(torch.equal(result, target_result))

    def pos_neg_contributions_conv_2d(self, baseline: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor,
                                      target_pos: torch.Tensor, target_neg: torch.Tensor):
        explainer = models.get_explainer(model_name="super_simple_convolutional_non_linear", baseline=baseline, shap=False)
        explainer.model[0][1].weight.data = weight
        explainer.set_diff_from_ref(input_tensor)
        pos, neg = non_linear.get_pos_neg_contributions_conv(dl=explainer, layer_name=explainer.model[0][0])
        pos = pos.cpu()
        neg = neg.cpu()
        self.assertTrue(torch.sum(torch.abs(neg - target_neg)) < 1e-5)
        self.assertTrue(torch.sum(torch.abs(pos - target_pos)) < 1e-5)

    def test_pos_neg_contributions_conv_2d(self):
        baselines = [
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32, requires_grad=True)
        ]
        input_tensors = [
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[-2, -1, 0], [1, 2, 3], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[2, 2, 2], [2, 2, 2], [2, 2, 2]]]], dtype=torch.float32, requires_grad=True),
        ]
        weights = [
            torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=True),
        ]
        target_pos = [
            torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[0, 0], [1, 2]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[2, 2], [2, 2]]]], dtype=torch.float32, requires_grad=True),
        ]
        target_neg = [
            torch.tensor([[[[0, 0], [0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[-1, -1], [-1, -1]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[-2, -1], [0, 0]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[-2, -2], [-2, -2]]]], dtype=torch.float32, requires_grad=True),
        ]
        for i, (base, inp, w, tp, tn) in enumerate(zip(baselines, input_tensors, weights, target_pos, target_neg)):
            self.pos_neg_contributions_conv_2d(base, inp, w, tp, tn)

    def pos_neg_contributions_conv_3d(self, baseline: torch.Tensor, input_tensor: torch.Tensor, weight: torch.Tensor,
                                      target_pos: torch.Tensor, target_neg: torch.Tensor):
        explainer = models.get_explainer(model_name="super_simple_convolutional_non_linear_3d", baseline=baseline, shap=False)
        explainer.model[0][1].weight.data = weight
        explainer.set_diff_from_ref(input_tensor)
        pos, neg = non_linear.get_pos_neg_contributions_conv(dl=explainer, layer_name=explainer.model[0][0])
        # print(f"baseline: {baseline}")
        # print(f"input_tensor: {input_tensor}")
        # print(f"weight: {weight}")
        # print(f"pos: {pos}")
        # print(f"target_pos: {target_pos}")
        # print(f"neg: {neg}")
        # print(f"target_neg: {target_neg}")
        pos = pos.cpu()
        neg = neg.cpu()
        self.assertTrue(torch.equal(pos, target_pos))
        self.assertTrue(torch.equal(neg, target_neg))

    def test_pos_neg_contributions_conv_3d(self):
        baseline = torch.tensor([[[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]], dtype=torch.float32, requires_grad=True)
        input_tensor = torch.tensor([[[[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
                                    dtype=torch.float32, requires_grad=True)
        weights = [
            torch.tensor([[[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[[1, 0], [0, 0]], [[0, 0], [0, -1]]]]], dtype=torch.float32, requires_grad=True),
        ]
        target_pos = [
            torch.tensor([[[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]], dtype=torch.float32, requires_grad=True),
        ]
        target_neg = [
            torch.tensor([[[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]], dtype=torch.float32, requires_grad=True),
            torch.tensor([[[[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]]]], dtype=torch.float32, requires_grad=True),
        ]
        for w, tp, tn in zip(weights, target_pos, target_neg):
            self.pos_neg_contributions_conv_3d(baseline, input_tensor, w, tp, tn)

    def test_flatten_pos_neg(self):
        baseline = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="simple_flatten_before_non_linear", baseline=baseline, shap=False)
        explainer.set_diff_from_ref(torch.tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=torch.float32,
                                                 requires_grad=True))
        neg = torch.zeros((1, 4, 4), dtype=torch.float32, requires_grad=True)
        pos = torch.zeros((1, 4, 4), dtype=torch.float32, requires_grad=True)
        unflattened_results = flatten.flatten_reverse(dl=explainer, shape=[1, 1, 2, 2], previous_multipliers=(pos, neg))
        self.assertEqual(unflattened_results[0].shape, (1, 4, 1, 2, 2))
        self.assertEqual(unflattened_results[1].shape, (1, 4, 1, 2, 2))

    def test_pos_neg_contribution_calc_input(self):
        baseline = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_ReLU_first", baseline=baseline, shap=False)
        explainer.set_diff_from_ref(torch.ones((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True))
        pos_explicit_input, neg_explicit_input = non_linear.get_pos_neg_contributions_input(dl=explainer)
        pos, neg = non_linear.get_pos_neg_contributions(dl=explainer, previous_layer_name="input",
                                                        current_layer_name=explainer.model[0][0])
        pos = pos.cpu()
        neg = neg.cpu()
        pos_explicit_input = pos_explicit_input.cpu()
        neg_explicit_input = neg_explicit_input.cpu()
        self.assertTrue(torch.equal(pos_explicit_input, pos))
        self.assertTrue(torch.equal(neg_explicit_input, neg))
        self.assertTrue(torch.equal(pos, torch.tensor([[[[[1., 1.]]]]])))
        self.assertTrue(torch.equal(neg, torch.tensor([[[[[0., 0.]]]]])))

    def test_pos_neg_contribution_calc_input_flatten(self):
        baseline = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_flatten_first", baseline=baseline, shap=False)
        explainer.set_diff_from_ref(torch.ones((1, 1, 1, 1, 2), dtype=torch.float32, requires_grad=True))
        pos, neg = non_linear.get_pos_neg_contributions(dl=explainer, previous_layer_name=explainer.model[0][0],
                                                        current_layer_name=explainer.model[1][0])
        pos = pos.cpu()
        neg = neg.cpu()
        self.assertTrue(torch.equal(pos, torch.tensor([[1., 1.]])))
        self.assertTrue(torch.equal(neg, torch.tensor([[0., 0.]])))

    def test_avg_pool_pos_neg(self):
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_avgpool", baseline=baseline, shap=False)
        # explanations, deltas = explainer.attribute(torch.tensor([[[[0., -1.], [-2., 3.]]]], dtype=torch.float32,
        #                                                         requires_grad=True),
        #                                            input_layer_of_interest=0, non_linearity_method="reveal_cancel")
        pos_multips = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
        neg_multips = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
        pos_multips = torch.reshape(pos_multips, (1, 1, 1, 1, 1))
        neg_multips = torch.reshape(neg_multips, (1, 1, 1, 1, 1))
        explainer.set_diff_from_ref(torch.tensor([[[[0., -1.], [-2., 3.]]]], dtype=torch.float32,
                                                 requires_grad=True))
        multipliers = avgpool.avgpool(dl=explainer, current_layer=explainer.model[0][1],
                                      prev_layer_diff=explainer.diff_from_ref["input"],
                                      previous_multipliers=(pos_multips, neg_multips))
        contributions = explainer.calculate_contributions(input_layer_of_interest="input", multipliers=multipliers)
        if torch.cuda.is_available():
            multipliers = multipliers.cpu()
            contributions = contributions.cpu()
        self.assertTrue(torch.equal(multipliers, torch.tensor([[[[[.125, .125], [.125, .125]]]]])))
        self.assertTrue(torch.equal(contributions, torch.tensor([[[[[0, -.125], [-.25, .375]]]]])))

    def test_pos_neg_contribution_calc_avgpool(self):
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_avgpool", baseline=baseline, shap=False)
        explainer.set_diff_from_ref(torch.tensor([[[[0, -1], [-2, 3]]]], dtype=torch.float32, requires_grad=True))
        pos, neg = non_linear.get_pos_neg_contributions(dl=explainer, previous_layer_name=explainer.model[0][0],
                                                        current_layer_name=explainer.model[1][0])
        pos = pos.cpu()
        neg = neg.cpu()
        self.assertTrue(torch.equal(pos, torch.tensor([[[[0.75]]]], dtype=torch.float32)))
        self.assertTrue(torch.equal(neg, torch.tensor([[[[-0.75]]]], dtype=torch.float32)))

    def test_pos_neg_contribution_calc_maxpool(self):
        # setup of random DeepLiftClass object, internals don't matter
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_maxpool", baseline=baseline, shap=False)
        # setting up the actual values to be tested
        explainer.diff_from_ref = {"test_case": torch.tensor([[[[0, -1], [-2, 3]]]], dtype=torch.float32,
                                                             requires_grad=True)}
        pos, neg = non_linear.get_pos_neg_contributions_maxpool(dl=explainer, layer_name="test_case")
        self.assertTrue(torch.equal(pos, torch.tensor([[[[0, 0], [0, 3]]]], dtype=torch.float32)))
        self.assertTrue(torch.equal(neg, torch.tensor([[[[0, -1], [-2, 0]]]], dtype=torch.float32)))

    def test_pos_neg_contribution_calc_non_linear(self):
        # setup of random DeepLiftClass object, internals don't matter
        baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="tiny_avgpool", baseline=baseline, shap=False)
        # setting up the actual values to be tested
        explainer.diff_from_ref = {"test_case": torch.tensor([[0, -1, -2, 3]], dtype=torch.float32,
                                                             requires_grad=True)}
        pos, neg = non_linear.get_pos_neg_contributions_non_linear(dl=explainer, layer_name="test_case")
        self.assertTrue(torch.equal(neg, torch.tensor([[0, -1, -2, 0]], dtype=torch.float32)))
        self.assertTrue(torch.equal(pos, torch.tensor([[0, 0, 0, 3]], dtype=torch.float32)))

    def test_multiplier_propagation_rescale_pos_neg(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="rescale_pos_neg_model", baseline=baseline, shap=False)
        input_values = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        explainer.set_diff_from_ref(input_values)
        pos = torch.tensor([[[1., 0.], [0, 0]]], dtype=torch.float32, requires_grad=True)
        neg = torch.tensor([[[.5, 0.], [0, 0]]], dtype=torch.float32, requires_grad=True)
        multipliers = non_linear.rescale_rule(dl=explainer, current_layer_name="1", previous_multipliers=(pos, neg))
        multipliers = multipliers.cpu()
        self.assertTrue(torch.equal(multipliers, torch.tensor([[[1., 0.], [0, 0]]], dtype=torch.float32)))

    def test_multiplier_propagation_reveal_cancel_pos_neg(self):
        baseline = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        explainer = models.get_explainer(model_name="rescale_pos_neg_model", baseline=baseline, shap=False)
        input_values = torch.tensor([[1, 1, 1]], dtype=torch.float32, requires_grad=True)
        explainer.set_diff_from_ref(input_values)
        pos_in = torch.tensor([[[1., 0.], [0, 0]]], dtype=torch.float32, requires_grad=True)
        neg_in = torch.tensor([[[.5, 0.], [0, 0]]], dtype=torch.float32, requires_grad=True)
        pos_out, neg_out = non_linear.reveal_cancel_rule(dl=explainer, current_layer_name="1",
                                                         previous_multipliers=(pos_in, neg_in))
        pos_target = torch.tensor([[[1., 0.], [0, 0]]], dtype=torch.float32)
        neg_target = torch.tensor([[[.25, 0.], [0, 0]]], dtype=torch.float32)
        pos_out = pos_out.cpu()
        neg_out = neg_out.cpu()
        self.assertTrue(torch.equal(pos_out, pos_target))
        self.assertTrue(torch.equal(neg_out, neg_target))

    # def test_final_contribution_calculation_pos_neg(self):
    #     # setup of random DeepLiftClass object, internals don't matter
    #     baseline = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    #     explainer = models.get_explainer(model_name="tiny_avgpool", baseline=baseline)
    #     # setting up the actual values to be tested
    #     explainer.diff_from_ref = {"test_case": torch.tensor([[0, -1, -2, 3]], dtype=torch.float32,
    #                                                          requires_grad=True)}
    #     explainer.model.layers = [("test_case", None), ("second_layer", None)]
    #     multipliers = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32, requires_grad=True)
    #     contrib_standard = explainer.calculate_contributions(input_layer_of_interest="second_layer",
    #                                                          multipliers=multipliers)
    #     multipliers_pos_neg = (multipliers, multipliers)
    #     contrib_pos_neg = explainer.calculate_contributions(input_layer_of_interest="second_layer",
    #                                                         multipliers=multipliers_pos_neg)
    #     print(contrib_standard)
    #     print(contrib_pos_neg)
    #     self.assertTrue(torch.equal(contrib_standard, contrib_pos_neg))


if __name__ == '__main__':
    unittest.main()
