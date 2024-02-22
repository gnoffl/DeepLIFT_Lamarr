import os
import pickle
import torch
import tests.regression_models as regression_models
import tests.classifier_models as classifier_models
import deeplift.deeplift as deeplift


name_to_function_dict = {
    "hidden_layer_classifier": classifier_models.get_hidden_layer_classifier,
    "linear_relu_min_example": regression_models.get_linear_relu_min_example,
    "MNIST_net_double_max": regression_models.get_real_MINST_net_double_max,
    "MNIST_net_mixed": regression_models.get_real_MINST_net_mixed,
    "MNIST_net_no_max": regression_models.get_real_MINST_net_no_max,
    "non_zero_padding": regression_models.get_non_zero_padding_net,
    "real_simple_model": regression_models.train_min_example,
    "rescale_pos_neg_model": regression_models.get_rescale_pos_neg_model,
    "sequential_linear": regression_models.get_bigger_sequential_fully_linear_trained,
    "sequential_non_linear": regression_models.get_bigger_sequential_non_linear_trained,
    "sequential_non_linear_random_weights": regression_models.get_sequential_non_linear_random_weights,
    "sequential_non_linear_random_weights_dropout": regression_models.get_sequential_non_linear_random_weights_dropout,
    "simple_flatten_before_non_linear": regression_models.get_tiny_flatten_before_non_linear,
    "simple_mixed_net": regression_models.get_convolution_into_linear_mixed,
    "simple_mixed_net_multiple_outputs": regression_models.get_convolution_into_linear_mixed_multiple_outputs,
    "simplest_classifier": classifier_models.get_simplest_classifier,
    "simplest_classifier_relu": classifier_models.get_simplest_classifier_relu,
    "small_conv_classifier": classifier_models.get_small_conv_classifier,
    "small_conv_classifier_easy_weights": classifier_models.get_small_conv_classifier_easy_weights,
    "small_linear_sigmoid": regression_models.get_small_linear_sigmoid,
    "super_short_2d_avg_pooling": regression_models.get_short_example_2d_avg_pooling,
    "super_simple_convolutional_1d_max_pooling": regression_models.get_convolutional_example_1d_max_pooling,
    "super_simple_convolutional_2d_avg_pooling": regression_models.get_convolutional_example_2d_avg_pooling,
    "super_simple_convolutional_2d_avg_pooling_multi_channel": regression_models.get_convolutional_example_2d_avg_pooling_multi_channel,
    "super_simple_convolutional_2d_max_pooling": regression_models.get_convolutional_example_2d_max_pooling,
    "super_simple_convolutional_2d_max_pooling_multi_channel": regression_models.get_convolutional_example_2d_max_pooling_multi_channel,
    "super_simple_convolutional_3d_max_pooling": regression_models.get_convolutional_example_3d_max_pooling,
    "super_simple_convolutional_linear": regression_models.get_convolutional_example_linear,
    "super_simple_convolutional_non_linear": regression_models.get_convolutional_example_non_linear,
    "super_simple_convolutional_non_linear_1d": regression_models.get_convolutional_example_non_linear_1d,
    "super_simple_convolutional_non_linear_3d": regression_models.get_convolutional_example_non_linear_3d,
    "super_simple_sequential_linear": regression_models.get_sequential_linear,
    "super_simple_sequential_linear_multiple_outputs": regression_models.get_sequential_linear_multiple_outputs,
    "super_simple_sequential_non_linear": regression_models.get_sequential_non_linear,
    "super_simple_sequential_non_linear_multiple_outputs": regression_models.get_sequential_non_linear_multiple_outputs,
    "tiny_avgpool": regression_models.create_tiny_avgpool,
    "tiny_flatten_first": regression_models.create_tiny_flatten_first,
    "tiny_linear_sigmoid": regression_models.get_tiny_linear_sigmoid,
    "tiny_maxpool": regression_models.create_tiny_maxpool,
    "tiny_ReLU_first": regression_models.create_tiny_ReLU_first,
}


def get_model(model_name: str, return_torch_script: bool = False) \
        -> torch.nn.Module or torch.jit._script.RecursiveScriptModule:
    """
    loads a model from the folder saved_networks. If the model does not exist, it is created and saved.

    Args:
        model_name: name of the model
        return_torch_script: determines whether the model should be returned as torchscript or the standard
            pytorch object

    Returns:
        the loaded or created model, as a torchscript or pytorch object
    """
    file_ending = "pt" if return_torch_script else "pkl"
    folder_path = os.path.join(os.path.dirname(__file__), "saved_networks")
    file_path = os.path.join(folder_path, f"{model_name}.{file_ending}")
    if not os.path.exists(file_path):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        creation_function = name_to_function_dict[model_name]
        model = creation_function()
        if return_torch_script:
            model = torch.jit.script(model)
    else:
        if return_torch_script:
            model = torch.jit.load(file_path)
        else:
            with open(file_path, 'rb') as file_path:
                model = pickle.load(file_path)
    return model


def get_explainer(model_name: str, baseline: torch.Tensor, shap: bool = True) -> deeplift.DeepLiftClass:
    """
    creates an explainer for a given model and baseline.

    Args:
        model_name: name of the model for the explainer
        baseline: reference input for the explainer
        shap: determines whether the standard DeepLIFT values or shap values should be calculated

    Returns:
        The created explainer
    """
    model = get_model(model_name, return_torch_script=True)
    return deeplift.DeepLiftClass(model=model, reference_value=baseline, shap=shap)


if __name__ == "__main__":
    l = [(key, value) for key, value in name_to_function_dict.items()]
    l.sort(key=lambda x: x[0].lower())
    for key, value in l:
        print(f"\"{key}\": {value.__name__},")
