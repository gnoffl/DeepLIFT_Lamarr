import math
from typing import Tuple
from sklearn.datasets import make_classification
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import tests.regression_models as regression_models


def create_data_folder():
    data_path = os.path.join(os.path.dirname(__file__), "data")
    classifier_path = os.path.join(data_path, "classification")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    if not os.path.isdir(classifier_path):
        os.mkdir(classifier_path)


def get_test_classification_dataset_linear(random_state=42, force_recreate=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    loads or creates a test dataset for classification

    Will save the dataset in
    Deeplift/Deeplift_new/Tests/data/classification/test_classification_dataset_{random_state}.pkl. If the dataset
    already exists, it will be loaded from there.

    Args:
        random_state: seed for the random number generator.
        force_recreate: determines whether the dataset should be recreated even if it already exists.

    Returns:
        tuple of the input and target data.
    """
    create_data_folder()
    data_path = os.path.join(os.path.dirname(__file__), "data")
    classifier_path = os.path.join(data_path, "classification")
    target_path = os.path.join(classifier_path, f"test_classification_dataset_{random_state}.pkl")
    if (not force_recreate) and os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            x, y = pickle.load(f)
    else:
        x, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1,
                                   n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                   flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                                   shuffle=True, random_state=random_state)
        with open(target_path, "wb") as f:
            pickle.dump((x, y), f)
    return x, y


def get_test_classification_dataset_conv(random_state: int = 42, shape: Tuple[int, int] = (3, 3),
                                         force_recreate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    loads or creates a test dataset for classification.

    Dataset is created with sklearn.datasets.make_classification and mimics image data for convolutional neural
    networks. Will save the dataset in
    Deeplift/Deeplift_new/Tests/data/classification/test_classification_conv_dataset_{random_state}.pkl. If the dataset
    already exists, it will be loaded from there.

    Args:
        random_state: seed for the random number generator.
        shape: shape of the data points.
        force_recreate: determines whether the dataset should be recreated even if it already exists.

    Returns:
        tuple of the input and target data.
    """
    create_data_folder()
    data_path = os.path.join(os.path.dirname(__file__), "data")
    classifier_path = os.path.join(data_path, "classification")
    target_path = os.path.join(classifier_path, f"test_classification_conv_dataset_{random_state}.pkl")
    if (not force_recreate) and os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            x, y = pickle.load(f)
    else:
        features = shape[0] * shape[1]
        informative = math.ceil(features / 2)
        x, y = make_classification(n_samples=1000, n_features=features, n_informative=informative, n_classes=2,
                                   n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True,
                                   shift=0.0, scale=1.0, shuffle=True, random_state=random_state)
        x = x.reshape(-1, 1, *shape)
        with open(target_path, "wb") as f:
            pickle.dump((x, y), f)
    return x, y


def train_model(net: nn.Module, dataset: Tuple[np.ndarray, np.ndarray], scale_outputs: bool = False):
    """
    trains a given model on a given dataset

    Args:
        net: the model to train
        dataset: the dataset to train on
        scale_outputs: determines whether the outputs should be scaled to [0, 1] before training
    """
    batchsize = 50
    x, y = dataset
    if not len(x) == len(y):
        raise ValueError("x and y must have the same length")
    train_len = int(0.9 * len(x))
    x_train, x_test = torch.tensor(x[:train_len], dtype=torch.float), torch.tensor(x[train_len:], dtype=torch.float)
    y_train, y_test = torch.tensor(y[:train_len], dtype=torch.float), torch.tensor(y[train_len:], dtype=torch.float)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    if scale_outputs:
        sigmoid = nn.Sigmoid()
    for epoch in range(100):
        for batch in range(0, train_len, batchsize):
            batch_input = x_train[batch:batch + batchsize]
            batch_target = y_train[batch:batch + batchsize]
            optimizer.zero_grad()
            outputs = net(batch_input)
            if scale_outputs:
                outputs = sigmoid(outputs)
            loss = criterion(outputs, batch_target.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")
    # test the net
    outputs = net(x_test)
    outputs = outputs.squeeze(1)
    predicted = (outputs > 0.5).float()
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    print(y_test)
    print('Accuracy of the network on the 100 test images: %d %%' % (100 * correct / total))


def get_simplest_classifier() -> nn.Module:
    """
    creates a simple classifier with one linear layer and a sigmoid activation function.

    The classifier is trained on a dataset created with sklearn.datasets.make_classification. The model is saved in
    Deeplift/Deeplift_new/Tests/saved_networks/simplest_classifier.pt. If the model already exists, it will be loaded
    from there.

    Returns:
        the created classifier.

    """
    net = nn.Sequential(
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    train_model(net=net, dataset=get_test_classification_dataset_linear())
    regression_models.save_model(net, "simplest_classifier")
    return net


def get_simplest_classifier_relu() -> nn.Module:
    """
    creates a simple classifier with one linear layer and a ReLU activation function.

    Returns:
        the created classifier.
    """
    net = nn.Sequential(
        nn.Linear(5, 1),
        nn.ReLU(),
    )
    train_model(net=net, dataset=get_test_classification_dataset_linear(), scale_outputs=True)
    regression_models.save_model(net, "simplest_classifier_relu")
    return net


def get_hidden_layer_classifier() -> nn.Module:
    """
    creates a simple classifier with one hidden layer with a ReLU activation function and sigmoid output layer.

    Returns:
        the created classifier.
    """
    net = nn.Sequential(
        nn.Linear(5, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )
    train_model(net=net, dataset=get_test_classification_dataset_linear())
    regression_models.save_model(net, "hidden_layer_classifier")
    return net


def get_small_conv_classifier() -> nn.Module:
    """
    creates a simple classifier with a convolution layer with a ReLU activation function and sigmoid output layer.

    Returns:
        the created classifier.
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, (2, 2), bias=True),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4, 1, bias=True),
        nn.Sigmoid()
    )
    dataset = get_test_classification_dataset_conv()
    train_model(net=net, dataset=dataset)
    regression_models.save_model(net, "small_conv_classifier")
    return net


def get_small_conv_classifier_easy_weights() -> nn.Module:
    """
    creates a simple classifier with a convolution layer with a ReLU activation function and sigmoid output layer.

    Returns:
        the created classifier.
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, (2, 2), bias=True),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4, 1, bias=True),
        nn.Sigmoid()
    )
    net[0].weight.data = torch.tensor([[[[-0.5, -1], [1.5, 2]]]], dtype=torch.float)
    net[0].bias.data = torch.tensor([0.5], dtype=torch.float)
    net[3].weight.data = torch.tensor([[0.5, -1, -0.5, 1]], dtype=torch.float)
    net[3].bias.data = torch.tensor([0.5], dtype=torch.float)
    regression_models.save_model(net, "small_conv_classifier_easy_weights")
    return net


if __name__ == '__main__':
    get_simplest_classifier()
