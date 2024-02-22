import os.path
from typing import Dict, List, Tuple
import src.deeplift.deeplift as dl
import tests.regression_models as regression_models
import tests.models as models
import src.deeplift.parsing as parsing
from captum.attr import DeepLift
import pickle
import torch
import math


class ComparisonResult:
    captum_attributions: torch.Tensor
    new_dl_attributions: torch.Tensor
    captum_deltas: torch.Tensor
    new_dl_deltas: torch.Tensor
    name: str
    multiplier: int


def get_model_path(model_name: str) -> str:
    model_jit_path = os.path.join(os.path.dirname(__file__), "saved_networks", model_name)
    return model_jit_path


def test_captum() -> None:
    """
    Test basic functionality of captum deeplift
    """
    model_name = "super_simple_sequential"
    models.get_model(model_name=model_name)
    model = torch.jit.load(get_model_path(f"{model_name}.pt"))
    capt_dl = DeepLift(model)
    inputs = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    baselines = torch.tensor([[2.5, 2.5, 2.5]], requires_grad=True)
    print(capt_dl.attribute(inputs=inputs, baselines=baselines))


def get_corresponding_functions_and_files() -> List[Dict[str, str]]:
    """
    Get a dictionary with the corresponding function names and file names for loading the super simple models

    Returns:
        dict with file name under "file" and function name under "function"
    """
    dicts = [
        {"file": "super_simple_sequential_linear.pt", "function": "get_sequential_linear"},
        {"file": "super_simple_sequential_non_linear.pt", "function": "get_sequential_non_linear"}
    ]
    return dicts


def do_comparison(baselines: torch.Tensor, captum_explainer: DeepLift, new_dl: dl.DeepLiftClass, inputs: torch.Tensor,
                  verbose: bool = False, target: int = None, efficient_target_calc: bool = True, delta: bool = False)\
        -> ComparisonResult:
    """
    Compares Captum implementation with the new implementation for the given setting

    Args:
        baselines: average value of the training set
        captum_explainer: deeplift explainer from captum
        new_dl: new deeplift explainer
        inputs: inputs to get the explanations for
        verbose: determines if only the difference is printed or also additional information
        target: target class for the explanation as index of the output vector
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards
        delta: determines if values related to the delta calculation should be printed

    Returns:
        the difference between the two implementations
    """
    target = target if target is not None else 0
    temp_target = target if efficient_target_calc else None
    later_indexing_necessary = False if efficient_target_calc else True
    new_dl_attributions, new_dl_deltas = new_dl.attribute(input_tensor=inputs, input_layer_of_interest='input',
                                                          non_linearity_method='rescale',
                                                          target_output_index=temp_target)
    if later_indexing_necessary:
        new_dl_attributions = new_dl_attributions[:, target]
        new_dl_deltas = new_dl_deltas[:, target]
    else:
        new_dl_attributions = new_dl_attributions[:, 0]
        new_dl_deltas = new_dl_deltas[:, 0]
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        baselines = baselines.cuda()
        captum_explainer.model.cuda()
    captum_attributions, captum_deltas = captum_explainer.attribute(inputs=inputs, baselines=baselines, target=target,
                                                                    return_convergence_delta=True)
    if torch.cuda.is_available():
        new_dl_attributions = new_dl_attributions.cpu()
        new_dl_deltas = new_dl_deltas.cpu()
        captum_attributions = captum_attributions.cpu()
        captum_deltas = captum_deltas.cpu()
    difference = captum_attributions - new_dl_attributions
    difference = torch.abs(difference)
    if verbose:
        print(f"input: {inputs}")
        output = new_dl.model.model(inputs)
        reference_output = new_dl.model.model(baselines)
        print(f"Output: {output}")
        print(f"Reference output: {reference_output}")
        print(f"Captum: {captum_attributions}")
        print(f"Captum shape: {captum_attributions.shape}")
        print(f"New: {new_dl_attributions}")
        print(f"New shape: {new_dl_attributions.shape}")
        print(f"deltas: {new_dl_deltas}")
        print(f"Difference: {difference}")
        print(f"Maximum difference is {torch.max(difference)}")
    if not verbose and delta:
        output = new_dl.model.model(inputs)
        reference_output = new_dl.model.model(baselines)
        print(f"Output: {output}")
        print(f"Reference output: {reference_output}")
        print(f"Contributions: {new_dl_attributions}")
        print(f"Delta: {new_dl_deltas}")
    # print warning if a value in the difference tensor is bigger than 0.001
    if torch.max(difference) > 0.001:
        print("Warning: Difference bigger than 0.001!!!!!\n\n")
    else:
        if verbose:
            print("\n")
    result = ComparisonResult()
    result.captum_attributions = captum_attributions
    result.new_dl_attributions = new_dl_attributions
    result.new_dl_deltas = new_dl_deltas
    result.captum_deltas = captum_deltas
    return result


def run_simple_comparisons(inputs: torch.Tensor = None, verbose: bool = False, delta: bool = False,
                           efficient_target_calc: bool = True) -> List[ComparisonResult]:
    """
    Run comparisons for the super simple models

    Args:
        inputs: input values to get the explanations for
        verbose: determines if only the difference is printed or also additional information
        delta: determines if values related to the delta calculation should be printed
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    dicts = get_corresponding_functions_and_files()
    results = []
    for dict_ in dicts:
        if verbose:
            print(f"super simple {' '.join(dict_['function'].split('_')[2:])}")
            print(f"input: {inputs}")
        captum_explainer = DeepLift(regression_models.__dict__[dict_["function"]]())
        baselines = torch.tensor([[2.5, 2.5, 2.5]], requires_grad=True)
        file_path = get_model_path(dict_["file"])
        model_name = file_path.split("/")[-1].split(".")[0]
        models.get_model(model_name=model_name)
        new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(torch.jit.load(file_path)), reference_value=baselines)
        if inputs is None:
            inputs = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        result = do_comparison(baselines=baselines, captum_explainer=captum_explainer, new_dl=new_dl, inputs=inputs,
                               verbose=verbose, efficient_target_calc=efficient_target_calc, delta=delta)
        result.multiplier = 42
        result.name = dict_["function"]
        results.append(result)
    return results


def run_comparison_real_linear(inputs: torch.Tensor, training_set_path: str, model_pkl: str, model_jit: str,
                               verbose: bool = False, delta: bool = False, efficient_target_calc: bool = True)\
        -> List[ComparisonResult]:
    """
    Run comparisons for small linear models

    Args:
        inputs: input values to get the explanations for
        training_set_path: path to the training set
        model_pkl: path to the model pickle file
        model_jit: path to the model torch script file
        verbose: determines if only the difference is printed or also additional information
        delta: determines if values related to the delta calculation should be printed
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    results = []
    file_name = model_jit.split(os.sep)[-1].split(".")[0]
    model_type = " ".join(file_name.split("_")[1:])
    models.get_model(model_name=file_name)
    if verbose:
        print(f"Sequential realistic {model_type}")
    with open(training_set_path, "rb") as f:
        training_set: regression_models.MinDataSetFCNN = pickle.load(f)
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)
    captum = DeepLift(model)
    baselines = training_set.average_input
    new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(torch.jit.load(model_jit)), reference_value=baselines)
    result = do_comparison(baselines=baselines, captum_explainer=captum, new_dl=new_dl, inputs=inputs, verbose=verbose,
                           efficient_target_calc=efficient_target_calc, delta=delta)
    result.name = file_name
    result.multiplier = 42
    results.append(result)

    return results


def run_additional_tests(multiplier: int, additional_tests: List[Tuple[str, int]] = None, verbose: bool = False,
                         batch: bool = True, efficient_target_calc: bool = True, delta: bool = False)\
        -> List[ComparisonResult]:
    """
    runs linear tests for given models and multipliers

    Args:
        multiplier: determines the multiplier for the standard input, which is just ones
        additional_tests: list of tuples (name, input_dim) with the name of the model and the dimension of the input
        verbose: determines how much information is printed
        batch: determines if the models will get a batch of inputs to calculate the explanations for
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards
        delta: determines if values related to the delta calculation should be printed

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    if additional_tests is None:
        return []
    for name, input_dim in additional_tests:
        jit_model, nn_model = load_models(name=name)
        batch_val = 4 if batch else 1
        base_input = torch.ones([batch_val, input_dim], requires_grad=True, dtype=torch.float32)
        true_input = multiplier * base_input
        baselines = torch.zeros([1, input_dim], requires_grad=True, dtype=torch.float32)
        captum_explainer = DeepLift(nn_model)
        new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(jit_model), reference_value=baselines)
        result = do_comparison(baselines=baselines, captum_explainer=captum_explainer, new_dl=new_dl, inputs=true_input,
                               verbose=verbose, target=0, efficient_target_calc=efficient_target_calc, delta=delta)
        result.name = name
        result.multiplier = multiplier
        return [result]


def non_convolution_comparisons(verbose: bool = False, batch: bool = True,
                                additional_tests: List[Tuple[str, int]] = None, efficient_target_calc: bool = True)\
        -> List[ComparisonResult]:
    """
    Run comparisons for the non-convolutional models

    Args:
        verbose: determines if only the difference between prediction of the different explainers is printed or also
            additional information
        batch: determines if the models will get a batch of inputs to calculate the explanations for
        additional_tests: infor for additional tests. Each Tuple (name, input_dim) in the List represents one test.
            name is the name of the model to be tested, input_dim is the dimension of the input for the model
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    if verbose:
        print("Non-convolutional comparisons")
    failed_tests = []
    for multiplier in [2, 1, -1]:
        if batch:
            small_base_tensor = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], requires_grad=True)
            large_base_tensor = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]], requires_grad=True)
        else:
            small_base_tensor = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
            large_base_tensor = torch.tensor([[1.0, 1.0, 1.0, 1., 1.]], requires_grad=True)
        small_input = multiplier * small_base_tensor
        large_input = multiplier * large_base_tensor
        failed_tests.extend(run_simple_comparisons(inputs=small_input, verbose=verbose,
                                                   efficient_target_calc=efficient_target_calc))
        linear_pkl = get_model_path("sequential_linear.pkl")
        linear_pt = get_model_path("sequential_linear.pt")
        linear_training = get_model_path("sequential_linear_training_set.pkl")
        non_linear_pkl = get_model_path("sequential_non_linear.pkl")
        non_linear_pt = get_model_path("sequential_non_linear.pt")
        non_linear_training = get_model_path("sequential_non_linear_training_set.pkl")
        failed_tests.extend(run_comparison_real_linear(
            large_input,
            training_set_path=linear_training,
            model_pkl=linear_pkl,
            model_jit=linear_pt,
            verbose=verbose, efficient_target_calc=efficient_target_calc
        ))
        failed_tests.extend(run_comparison_real_linear(
            large_input,
            training_set_path=non_linear_training,
            model_pkl=non_linear_pkl,
            model_jit=non_linear_pt,
            verbose=verbose, efficient_target_calc=efficient_target_calc
        ))
        failed_tests.extend(run_additional_tests(additional_tests=additional_tests, verbose=verbose, batch=batch,
                                                 multiplier=multiplier, efficient_target_calc=efficient_target_calc))
        return failed_tests


def load_models(name: str) -> Tuple[torch.jit.ScriptModule, torch.nn.Module]:
    """
    Load the model from the given path or name

    Args:
        name: name of the model in Deeplift_new/saved_networks

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    model_jit_path = get_model_path(f"{name}.pt")
    model_pkl_path = get_model_path(f"{name}.pkl")
    models.get_model(model_name=name)
    with open(model_pkl_path, "rb") as f:
        nn_model = pickle.load(f)
    jit_model = torch.jit.load(model_jit_path)
    return jit_model, nn_model


def convolution_comparisons(test_parameters: List[Tuple[int, int, str]], verbose: bool = False, batch: bool = True,
                            efficient_target_calc: bool = True, delta: bool = False)\
        -> List[ComparisonResult]:
    """
    Run comparisons for the non-convolutional models

    Args:
        test_parameters: list of tuples containing information on the models to be tested. Each entry in the list
            corresponds to one model. The Tuple (dims, input_size, name) contains the number of dimensions of the model
            dims, the expected input size of the model (assumed to be input_size in all dimensions) and the name of the
            model to be loaded
        verbose: determines if only the difference between prediction of the different explainers is printed or also
            additional information
        batch: determines if the models will get a batch of inputs to calculate the explanations for
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
            efficient, or only applied afterwards
        delta: determines if values related to the delta calculation should be printed

    Returns:
        a list containing the tuples (function, multiplier), where function is the function name to load the model and
        multiplier is 42 (multiplier is only there to fit format, but contains no valuable information in this case)
    """
    if verbose:
        print("Convolutional comparisons")
    results = []
    for (dims, input_size, name) in test_parameters:
        jit_model, nn_model = load_models(name=name)
        if verbose:
            print(f"{name}")
        shape = [1, 1]
        for _ in range(dims):
            shape.append(input_size)
        base_input = torch.ones(shape, requires_grad=True, dtype=torch.float32)
        baselines = torch.zeros(shape, requires_grad=True, dtype=torch.float32)
        if batch:
            base_input = torch.cat([base_input, 2 * base_input[0:1]], 0)
            base_input = torch.cat([base_input, -1 * base_input[0:1]], 0)
            base_input = torch.cat([base_input, .5 * base_input[0:1]], 0)
            result = do_comparison(baselines=baselines, captum_explainer=DeepLift(nn_model),
                                   new_dl=dl.DeepLiftClass(parsing.SequentialLoadedModel(jit_model),
                                                           reference_value=baselines), inputs=base_input, verbose=verbose, target=0,
                                   efficient_target_calc=efficient_target_calc, delta=delta)
            result.name = name
            result.multiplier = int(batch)
            results.append(result)
        else:
            for multiplier in [2, 1, -1]:
                conv_input = multiplier * base_input
                captum_explainer = DeepLift(nn_model)
                new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(jit_model),
                                          reference_value=baselines)

                result = do_comparison(baselines=baselines, captum_explainer=captum_explainer, new_dl=new_dl,
                                       inputs=conv_input, verbose=verbose, target=0,
                                       efficient_target_calc=efficient_target_calc, delta=delta)
                result.name = name
                result.multiplier = multiplier
                results.append(result)
        return results


def print_failed_tests(failed_tests: List[ComparisonResult]):
    """
    prints the list of failed tests with the name of the test and the multiplier used.

    Args:
        failed_tests: list of failed tests with the multipliers
    """
    failed_counter = 0
    for result_obj in failed_tests:
        result = torch.abs(result_obj.captum_attributions - result_obj.new_dl_attributions)
        if torch.max(result) > 0.001:
            print(f"Test {result_obj.name} with multiplier {result_obj.multiplier} failed")
            failed_counter += 1
    if failed_counter:
        print(f"{failed_counter} tests failed")
    else:
        print("All tests positive")


def get_MNIST_avgs(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    calculates the average input for the MNIST dataset
    Args:
        loader: dataloader for the MNIST dataset

    Returns:
        the average input as a Tensor
    """
    save_path = os.path.join(os.path.dirname(__file__), "..", "Tests", "saved_networks", "MNIST_avgs.pkl")
    if not os.path.exists(save_path):
        for i, (data, target) in enumerate(loader):
            if i == 0:
                start_shape = [1]
                start_shape.extend(data.shape[1:])
                avgs = torch.zeros(start_shape)
            data_sum = data.sum(dim=0, keepdim=True)
            avgs += data_sum
        avgs /= len(loader.dataset)
        with open(save_path, "wb") as f:
            pickle.dump(avgs, f)
    else:
        with open(save_path, "rb") as f:
            avgs = pickle.load(f)
    return avgs


def compare_MNIST(verbose=False, batch=True, efficient_target_calc=False, delta: bool = False)\
        -> List[ComparisonResult]:
    """
    compares the captum implementation with the new implementation of deeplift for a model on the MNIST dataset

    Args:
        verbose: determines if there should be debugging information printed out
        batch: determines if the input for the model contains a single value or batched data
        efficient_target_calc: determines whether the results are computed efficiently or not
        delta: determines if values related to the delta calculation should be printed

    Returns:
        a list with zero or one elements. If the test passes, an empty list will be returned, otherwise the list will
        contain the tuple ("MNIST_net", 42)
    """
    return_list = []
    model_jit_path = get_model_path("MNIST_net_mixed.pt")
    test_loader, train_loader = regression_models.get_MNIST_dataloaders()
    baseline = get_MNIST_avgs(test_loader)
    for images, _ in train_loader:
        if batch:
            image = images[0:4]
        else:
            image = images[0:1]
        break
    model = regression_models.get_real_MNIST_net(lr=0.0005, num_epochs=2, batch_size=64, retrain=False)
    captum_explainer = DeepLift(model)
    new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(torch.jit.load(model_jit_path)), reference_value=baseline)

    result = do_comparison(baselines=baseline, captum_explainer=captum_explainer, new_dl=new_dl, inputs=image,
                           verbose=verbose, target=1, delta=delta, efficient_target_calc=efficient_target_calc)
    result.name = "MNIST_net"
    result.multiplier = 42
    return_list.append(result)
    return return_list


def test_sigmoid(verbose: bool = False, delta: bool = False) -> List[ComparisonResult]:
    """
    tests the deeplift implementation of the sigmoid layer by using do_comparison on models containing a sigmoid layer.

    Args:
        verbose: determines whether debugging info will be printed

    Returns:
        a list with zero or one elements. If the test passes, an empty list will be returned, otherwise the list will
        contain the tuple ("MNIST_net", 42)

    """
    input_ = torch.tensor([[math.log(2, math.e), math.log(2, math.e)]], dtype=torch.float, requires_grad=True)
    baseline = torch.tensor([[0.0, 0.0]], dtype=torch.float)
    for name in ["small_linear_sigmoid", "tiny_linear_sigmoid"]:
        if verbose:
            print(name)
        models.get_model(model_name=name)
        model_jit_path = get_model_path(f"{name}.pt")
        model_pkl_path = get_model_path(f"{name}.pkl")
        with open(model_pkl_path, "rb") as f:
            model = pickle.load(f)
        captum_explainer = DeepLift(model)
        jit_model = torch.jit.load(model_jit_path)
        new_dl = dl.DeepLiftClass(parsing.SequentialLoadedModel(jit_model), reference_value=baseline)
        result = do_comparison(baselines=baseline, captum_explainer=captum_explainer, new_dl=new_dl, inputs=input_,
                               verbose=verbose, delta=delta)
        result.name = "sigmoid"
        result.multiplier = 42
    return [result]


def standard_test(verbose: bool = False, efficient_target_calc: bool = False) -> None:
    """
    Run multiple integration tests to check if the new deeplift implementation works as expected

    Args:
        verbose: determines if only the difference between prediction of the different explainers is printed or also
        additional information
        efficient_target_calc: determines if the target class should be used for the calculation to make it run more
        efficient, or only applied afterwards
    """
    results: List[ComparisonResult] = []
    further_tests = [
        ("super_simple_sequential_linear_multiple_outputs", 3),
        ("super_simple_sequential_non_linear_multiple_outputs", 3)
    ]
    results.extend(non_convolution_comparisons(verbose=verbose, batch=True, additional_tests=further_tests,
                                               efficient_target_calc=efficient_target_calc))
    conv_nets = [
        (1, 3, "super_simple_convolutional_non_linear_1d"),
        (2, 3, "super_simple_convolutional_non_linear"),
        (3, 3, "super_simple_convolutional_non_linear_3d"),
        (2, 3, "simple_mixed_net_multiple_outputs"),
        (2, 3, "super_simple_convolutional_linear"),
        (1, 5, "super_simple_convolutional_1d_max_pooling"),
        (2, 5, "super_simple_convolutional_2d_max_pooling"),
        (3, 5, "super_simple_convolutional_3d_max_pooling"),
        (2, 5, "super_simple_convolutional_2d_max_pooling_multi_channel"),
        (2, 5, "super_simple_convolutional_2d_avg_pooling_multi_channel"),
        (2, 5, "super_simple_convolutional_2d_avg_pooling"),
        (2, 4, "super_short_2d_avg_pooling"),
    ]
    results.extend(convolution_comparisons(conv_nets, verbose=verbose, batch=True,
                                           efficient_target_calc=efficient_target_calc))

    results.extend(test_sigmoid(verbose=verbose))
    results.extend(compare_MNIST(verbose=verbose, batch=True, efficient_target_calc=efficient_target_calc))
    print_failed_tests(results)


def see_delta_propagation_MNIST(model_name: str = "MNIST_net"):
    test_loader, train_loader = regression_models.get_MNIST_dataloaders()
    baseline = get_MNIST_avgs(test_loader)
    for images, _ in train_loader:
        image = images[0:4]
        break
    models.get_model(model_name=model_name)
    model = torch.jit.load(get_model_path(f"{model_name}.pt"))
    dl_explainer = dl.DeepLiftClass(parsing.SequentialLoadedModel(model), baseline)
    for i in range(len(dl_explainer.model.get_all_layer_names()) - 1):
        print(dl_explainer.model[i][0])
        explanations, deltas = dl_explainer.attribute(image, input_layer_of_interest=i)
        print(f"sum of deltas: {torch.sum(deltas)}")
    print("\n")


def inspect_deltas_MNIST(verbose: bool = False, delta: bool = False):
    name, multiplier, captum, new_dl_attr_efficient, deltas = compare_MNIST(verbose=verbose, batch=True, delta=delta,
                                                                            efficient_target_calc=True)[0]
    #name, multiplier, captum, new_dl_attr_inefficient, deltas = compare_MNIST(verbose=verbose, batch=True, delta=delta,
    #                                                                           efficient_target_calc=False)[0]
    print(f"sum of deltas: {torch.sum(deltas)}")
    print(f"sum of differences: {torch.sum(torch.abs(captum - new_dl_attr_efficient))}")

    # res = []
    # for i in range(10):
    #     res.extend(compare_MNIST(efficient_target_calc=True, verbose=True))
    # for i, (name, multiplier, captum, new_dl_attr, deltas) in enumerate(res):
    #     print(f"delta calculated from layer {i}: {deltas}")


def show_max_pool_influence_deltas():
    names = ["MNIST_net_no_max", "MNIST_net_mixed", "MNIST_net_double_max"]
    for name in names:
        print(name)
        see_delta_propagation_MNIST(name)


if __name__ == "__main__":
    # standard_test(verbose=False, efficient_target_calc=False)
    # inspect_deltas_MNIST()
    show_max_pool_influence_deltas()
