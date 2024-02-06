import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import pickle
import os.path
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np


class LinearMNISTNet(nn.Module):
    """
    class that implements a simple linear neural network for the MNIST dataset
    """
    def __init__(self, input_size=28*28, hidden_size=500, num_classes=10) -> None:
        """
        initializes the network
        
        Args:
            input_size: number of input neurons for the network
            hidden_size: number of neurons in the single linear hidden layer
            num_classes: number of output neurons
        """
        super(LinearMNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the network
        
        Args:
            x: input data

        Returns:
            output of the network as a tensor
        """
        out = self.fc1(x)
        out = self.drop(out)
        out = self.fc2(out)
        return out


def get_MNIST_dataloaders(batch_size=100) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    get dataloaders for the MNIST dataset
    
    creates two torch dataloaders for the MNIST dataset, one for the training data and one for the testing data. The
    loader for the training data shuffles the data and the loader for the testing data does not shuffle the data.
    
    Args:
        batch_size: The batch size for the dataloaders

    Returns:
        a Tuple (train, test), where train is the dataloader for the training data and test is the dataloader for the
        testing data
    """
    train_data = torchvision.datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "data"), train=True,
                                            transform=transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "data"), train=False,
                                           transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_linear_net(train_loader, test_loader, num_epochs=20, lr=1e-3, update_interval=100) -> LinearMNISTNet:
    """
    trains a simple linear neural network for the MNIST dataset
    
    Args:
        train_loader: a torch data loader for the training data
        test_loader: a torch data loader for the testing data
        num_epochs: number of epochs to train the network
        lr: learning rate for the optimizer of the network
        update_interval: determines after how many batches an update of the current state of the training process will
        be printed

    Returns:
        the trained network
    """
    net = LinearMNISTNet()
    if torch.cuda.is_available():
        net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            if torch.cuda.is_available():
                images.cuda()
                labels.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % update_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step[{(i+1)}/{len(train_loader)}], loss: {loss.item()}")

        net.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28 * 28)
            if torch.cuda.is_available():
                images.cuda()
                labels.cuda()
            outputs = net(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds = (predictions == labels)
            total += predictions.size()[0]
            correct += correct_preds.sum()
        print(f"end of epoch {epoch+1}, correct predictions: {correct/total}")

    return net


def train_real_MNIST_net(lr: float, num_epochs: int, batch_size: int, nr_max_pool: int = 1) -> nn.Module:
    """
    trains a MNIST net containing different layer types to test the deeplift algorithm
    
        lr: learning rate
        num_epochs: number of epochs to train
        batch_size: batch size for training

    Returns:
        the trained net
    """
    if nr_max_pool == 0:
        net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )
    elif nr_max_pool == 1:
        net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )
    elif nr_max_pool == 2:
        net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )
    else:
        raise ValueError("nr_max_pool must be 0, 1 or 2")

    train_loader, test_loader = get_MNIST_dataloaders(batch_size=batch_size)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step[{(i+1)}/{len(train_loader)}], loss: {loss.item()}")

        net.eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds = (predictions == labels)
            total += predictions.size()[0]
            correct += correct_preds.sum()
        print(f"end of epoch {epoch+1}, correct predictions: {correct/total}")
    return net


def get_real_MNIST_net(lr: float = 0.0005, num_epochs: int = 2, batch_size: int = 64, retrain: bool = False,
                       nr_max_pool: int = 1) -> nn.Module:
    """
    loads a saved MNIST net if it exists, otherwise trains and saves a new one

    Args:
        lr: learning rate for training the net
        num_epochs: number of epochs to train the net
        batch_size: batch size for training the net
        retrain: determines if the net should be retrained even if a version was already saved
        nr_max_pool: determines whether a max pool layer should be present in the net.

    Returns:
        the trained network
    """
    if nr_max_pool == 0:
        addition = "_no_max"
    elif nr_max_pool == 1:
        addition = "_mixed"
    elif nr_max_pool == 2:
        addition = "_double_max"
    else:
        raise ValueError("nr_max_pool must be 0, 1 or 2")
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "saved_networks", f"MNIST_net{addition}.pkl")
    print(file_path)
    if not os.path.isfile(file_path) or retrain:
        net = train_real_MNIST_net(lr=lr, num_epochs=num_epochs, batch_size=batch_size, nr_max_pool=nr_max_pool)
        save_model(net, f"MNIST_net{addition}")
    else:
        with open(file_path, "rb") as f:
            net = pickle.load(f)
    return net


def get_real_MINST_net_no_max():
    """
    wrapper function for get_real_MNIST_net with nr_max_pool = 0

    Returns:
        the trained network
    """
    return get_real_MNIST_net(nr_max_pool=0)


def get_real_MINST_net_mixed():
    """
    wrapper function for get_real_MNIST_net with nr_max_pool = 1

    Returns:
        the trained network
    """
    return get_real_MNIST_net(nr_max_pool=1)


def get_real_MINST_net_double_max():
    """
    wrapper function for get_real_MNIST_net with nr_max_pool = 2

    Returns:
        the trained network
    """
    return get_real_MNIST_net(nr_max_pool=2)


class MinModel(nn.Module):
    """
    class that implements a neural network with 2 input nodes, 2 hidden nodes and 1 output node
    """
    def __init__(self) -> None:
        """
        initialize network
        """
        super().__init__()
        self.linear1 = nn.Linear(2, 2, bias=False)
        self.linear2 = nn.Linear(2, 1, bias=False)

    def forward(self, x) -> torch.Tensor:
        """
        forward pass of the network

        Args:
            x: input data

        Returns:
            output of the network
        """
        out = self.linear1(x)
        return self.linear2(out)


class SuperMinModel(nn.Module):
    """
    class that implements a neural network with 3 input nodes, 2 hidden nodes and 1 output node and given weights

    the weights are set in a way that the output is the sum of the input values
    """
    def __init__(self) -> None:
        """
        initialize network so that it adds the input values for the output
        """
        super().__init__()
        self.linear1 = nn.Linear(3, 2, bias=False)
        self.linear2 = nn.Linear(2, 1, bias=False)
        self.linear1.weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
        self.linear2.weight.data = torch.tensor([[1, -1]], dtype=torch.float)

    def forward(self, x) -> torch.Tensor:
        """
        forward pass of the network

        Args:
            x: input data

        Returns:
            output of the network
        """
        out = self.linear1(x)
        out = self.linear2(out)
        return out


class SuperMinModelNonLinear(nn.Module):
    """
    class that implements a neural small network with given weights and ReLU as activation function

    the weights are set in a way that the output is the sum of the input values for positive inputs.
    """
    def __init__(self) -> None:
        """
        initialize network so that it adds the input values for the output for pos values
        """
        super().__init__()
        self.linear1 = nn.Linear(3, 2, bias=False)
        self.linear2 = nn.Linear(2, 1, bias=False)
        self.linear1.weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
        self.linear2.weight.data = torch.tensor([[1, -1]], dtype=torch.float)

    def forward(self, x) -> torch.Tensor:
        """
        forward pass of the network

        Args:
            x: input data

        Returns:
            output of the network
        """
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        return out


class MinDataSetFCNN(pt.utils.data.Dataset):
    """
    generic dataset with given number of input values and single target value.

    entries can be given or randomly generated. If entries are generated, the target value is the sum of the input.
    """
    size: int
    nr_of_inputs: int
    average_input: pt.Tensor
    y: List[float]

    def __init__(self, size: int, nr_of_inputs: int = 2, values: List[Tuple] = None) -> None:
        """
        initializes the dataset

        if "values" argument is given, the dataset is initialized with the given values. otherwise, random values are
        generated.

        Args:
            size: number of samples in the dataset
            values: optional list of tuples containing the input and target values as tuples, where the last value is
                the target and the other values are the input values. if not provided, random values are generated
            nr_of_inputs: number of input values per data point
        """
        if values is not None:
            if len(values) != size:
                raise ValueError("Size of values must be equal to size")
            if len(values[0]) != nr_of_inputs + 1:
                raise ValueError("Each value must have a length of dimensions + 1")
            #create an attribute for the class for each dimension
            for i in range(nr_of_inputs):
                #create an attribute for the class in each iteration of the loop
                setattr(self, f"x{i}", [x[i] for x in values])
            self.y = [x[-1] for x in values]
        else:
            for i in range(nr_of_inputs):
                # create an attribute for the class in each iteration of the loop
                # the attribute is a list of random numbers between 0 and 1
                setattr(self, f"x{i}", [float(x) for x in np.random.uniform(-3, 3, size)])
            #create the target values as the sum of the input values plus some noise using a for loop
            for i in range(nr_of_inputs):
                if i == 0:
                    self.y = getattr(self, f"x{i}")
                else:
                    self.y = [self.y[j] + getattr(self, f"x{i}")[j] for j in range(size)]
                self.y = [self.y[i] + np.random.normal(0, 0.1) for i in range(size)]

        self.size = size
        self.nr_of_inputs = nr_of_inputs
        self.average_input = pt.tensor([[float(np.mean(getattr(self, f"x{i}"))) for i in range(nr_of_inputs)]])

    def __len__(self) -> int:
        """
        gives the number of entries of the dataset

        Returns:
            size of the dataset
        """
        return self.size

    def __getitem__(self, index: int) -> Tuple[pt.tensor, pt.tensor]:
        """
        gets the entry at the given index

        Args:
            index: index of the values to return
        Returns:
            a tuple containing the input values and the target value as tensors
        """
        #for each dimension, get the value at the index item and add it to a list
        #then return the list as a tensor
        values = pt.tensor([getattr(self, f"x{i}")[index] for i in range(self.nr_of_inputs)], dtype=pt.float)
        target = pt.tensor(self.y[index], dtype=pt.float)
        return values, target

    def get_average_input(self) -> pt.tensor:
        """
        gets the average input values of the dataset

        Returns:
            the average input values as a tensor
        """
        return self.average_input


class MinDataSetConv(pt.utils.data.Dataset):
    """
    generic dataset for convolutional neural networks.

    The shape of the input images is flexible. Entries of the dataset can be given or randomly generated.
    """
    size: int                       #number of samples in the dataset
    shape: Tuple[int, int]          #shape of the input images
    average_input: pt.Tensor        #tensor containing the averages of the input images
    x: pt.Tensor                    #tensor containing the input images
    y: pt.Tensor                    #tensor containing the target values

    def __init__(self, size: int, shape: Tuple, values: Tuple[torch.Tensor, torch.Tensor] = None) -> None:
        """
        initialize the dataset

        if "values" argument is given, the dataset is initialized with the given values. otherwise, random values of the
        given shape are generated.

        Args:
            size: number of samples in the dataset
            shape: shape of the input images
            values: optional list of tuples containing the input and target values, where the last value is
                the target and the other values are the input values. if not provided, random values are generated
        """
        list_ = [size]
        list_.extend(shape)
        full_shape = tuple(list_)
        if values is not None:
            if values[0].shape != full_shape:
                raise ValueError("shape of values must be equal to size x shape")
            if values[1].shape != (size,):
                raise ValueError("number of target values must be equal to size")
            self.x = values[0]
            self.y = values[1]
        else:
            self.x = pt.tensor(np.random.uniform(0, 1, full_shape), dtype=pt.float)
            self.y = pt.tensor(np.random.uniform(0, 1, (size,)), dtype=pt.float)

        self.size = size
        self.shape = shape
        self.average_input = self.x.mean(dim=0).reshape(1, *shape)

    def __len__(self) -> int:
        """
        gets the number of entries of the dataset

        Returns:
            size of the dataset
        """
        return self.size

    def __getitem__(self, index: int) -> Tuple[pt.tensor, pt.tensor]:
        """
        returns input and target values at the given index

        Args:
            index: index of the values to return

        Returns:
            Tuple (input, target) with input being the input image and target being the target value
        """
        return self.x[index], self.y[index]


def get_test_MNIST():
    """
    tensorflow implementation of a neural network for the MNIST dataset

    Returns:
        the trained model and the test data
    """
    mnist = tf.keras.datasets.mnist
    test = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = test
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if os.path.isfile(r"C:\Users\gerno\OneDrive\Desktop\model.h5"):
        model = tf.keras.models.load_model(r"C:\Users\gerno\OneDrive\Desktop\model.h5")
    else:

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)

    model.summary()
    return model, x_test


def train_min_example() -> MinModel:
    """
    trains a MinModel on a MinDataSetFCNN

    Returns:
        the trained model
    """
    model = MinModel()
    training_set = MinDataSetFCNN(2000)
    validation_set = MinDataSetFCNN(100)
    training_loader = pt.utils.data.DataLoader(training_set, batch_size=8, shuffle=True)
    validation_loader = pt.utils.data.DataLoader(validation_set, batch_size=8, shuffle=False)
    loss_fn = nn.MSELoss()
    optim = pt.optim.SGD(model.parameters(), lr=0.01)

    running_loss = 0.
    last_loss = 0.

    for i, data_ in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data_
        labels.unsqueeze_(1)
        # Zero your gradients for every batch!
        optim.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optim.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    max_i = 0
    val_loss = 0
    for i, data_ in enumerate(validation_loader):
        max_i = i
        val_inputs, val_labels = data_

        val_outputs = model(val_inputs)

        # Compute the loss and its gradients
        val_loss += loss_fn(val_outputs, val_labels)

    print(f"val_loss: {val_loss/max_i}")
    save_model(name="real_simple_model", model=model)
    with open(os.path.join("saved_networks", "real_simple_model_training_set.pkl"), 'wb') as f:
        pickle.dump(training_set, f)

    return model


def load_pkl_sequential(model_name: str) -> nn.Sequential:
    with open(os.path.join(os.path.dirname(__file__), "saved_networks", f"{model_name}.pkl"), 'rb') as f:
        model = pickle.load(f)
    return model


def get_linear_relu_min_example(redo: bool = False) -> nn.Sequential:
    if not redo and os.path.isfile(os.path.join(os.path.dirname(__file__), "saved_networks", "linear_relu_min_example.pkl")):
        return load_pkl_sequential("linear_relu_min_example")
    else:
        model= nn.Sequential(
            nn.Linear(3, 2, bias=False),
            nn.ReLU()
        )
        model[0].weight.data = torch.tensor([[1, -1, 1], [-1, 1, 0]], dtype=torch.float)
        save_model(model, "linear_relu_min_example")
        return model


# create sequential model copying the functionality of SuperMinModelNonLinear
def get_sequential_non_linear() -> nn.Sequential:
    """
    creates a simple 3 layer neural network with a single output and ReLU as activation function.

    model contains 3 input nodes, 2 hidden nodes and 1 output node. the weights are set in a way that the output is the
    sum of the input values for positive inputs. Model is saved as pickle and torchscript under the name
    super_simple_sequential_non_linear in the folder saved_networks.

    Returns:
        the model with the given weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 1, bias=False),
        nn.ReLU()
    )
    model[0].weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
    model[2].weight.data = torch.tensor([[1, -1]], dtype=torch.float)
    save_model(model, "super_simple_sequential_non_linear")
    return model


def get_sequential_non_linear_random_weights() -> nn.Sequential:
    """
    creates a simple 3 layer neural network with a single output and ReLU as activation function.

    model contains 3 input nodes, 2 hidden nodes and 1 output node. Model is saved as pickle and torchscript under the
    name super_simple_sequential_non_linear_random_weights in the folder saved_networks.

    Returns:
        the model with random weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.ReLU()
    )
    model.eval()
    save_model(model, "super_simple_sequential_non_linear_random_weights")
    return model


def get_sequential_non_linear_random_weights_dropout() -> nn.Sequential:
    """
    creates a simple 4 layer neural network with a single output and ReLU as activation function.

    model contains 3 input nodes, 2 hidden nodes and 1 output node. Model is saved as pickle and torchscript under the
    name super_simple_sequential_non_linear_random_weights_dropout in the folder saved_networks.

    Returns:
        the model with random weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2, 1),
        nn.ReLU()
    )
    model.eval()
    save_model(model, "super_simple_sequential_non_linear_random_weights_dropout")
    return model


# create sequential model copying the functionality of SuperMinModelNonLinear
def get_sequential_non_linear_multiple_outputs() -> nn.Sequential:
    """
    creates a simple 3 layer neural network with multiple outputs and ReLU as activation function.

    model contains 3 input nodes, 2 hidden nodes and 2 output nodes. The weights are set in a way that the output is the
    sum of the input values and the negative sum of the input values for positive inputs. Model is saved as pickle and
    torchscript under the name super_simple_sequential_non_linear_multiple_outputs in the folder saved_networks.

    Returns:
        the model with the given weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 2, bias=False),
        nn.ReLU()
    )
    model[0].weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
    model[2].weight.data = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float)
    save_model(model, "super_simple_sequential_non_linear_multiple_outputs")
    return model


def get_sequential_linear() -> nn.Sequential:
    """
    creates a simple 3 layer neural network with a single outputs.

    model contains 3 input nodes, 2 hidden nodes and 1 output node. The weights are set in a way that the output is the
    sum of the input values. Model is saved as pickle and torchscript under the name super_simple_sequential_linear in
    the folder saved_networks.

    Returns:
        the model with the given weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.Linear(2, 1, bias=False)
    )
    model[0].weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
    model[1].weight.data = torch.tensor([[1, -1]], dtype=torch.float)
    save_model(model, "super_simple_sequential_linear")
    return model


def get_sequential_linear_multiple_outputs() -> nn.Sequential:
    """
    creates a simple 3 layer neural network with multiple outputs.

    model contains 3 input nodes, 2 hidden nodes and 2 output nodes. The weights are set in a way that the output is the
    sum of the input values and the negative sum of the input values for positive inputs. Model is saved as pickle and
    torchscript under the name super_simple_sequential_linear_multiple_outputs in the folder saved_networks.

    Returns:
        the model with the given weights.
    """
    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.Linear(2, 2, bias=False)
    )
    model[0].weight.data = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.float)
    model[1].weight.data = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float)
    save_model(model, "super_simple_sequential_linear_multiple_outputs")
    return model


def get_bigger_sequential_linear_trained(training_set: MinDataSetFCNN = None, relu: bool = False) -> Tuple[nn.Sequential, MinDataSetFCNN]:
    """
    create and train a small sequential linear model.

    Whether the net has ReLU as activation function or not can be set with the relu argument. If no training set is
    given, a new one is generated and therefore the model is trained to add the inputs (default implementation of
    MinDataSetFCNN). The model is saved as pickle and torchscript under the name sequential_linear or
    sequential_non_linear in the folder saved_networks depending on the relu parameter.

    Returns:
        the trained model
    """
    if relu:
        net = nn.Sequential(
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            # nn.ReLU()
        )
    else:
        net = nn.Sequential(
            nn.Linear(5, 4),
            nn.Linear(4, 3),
            nn.Linear(3, 1)
        )
    #train net using a MinDataSet
    net.train()
    if not training_set:
        training_set = MinDataSetFCNN(size=500, nr_of_inputs=5)
    val_set = MinDataSetFCNN(size=100, nr_of_inputs=5)
    training_loader = pt.utils.data.DataLoader(training_set, batch_size=8, shuffle=True)
    val_loader = pt.utils.data.DataLoader(val_set, batch_size=8, shuffle=False)
    loss_fn = nn.MSELoss()
    optim = pt.optim.Adam(net.parameters(), lr=0.001)
    for j in range(20):
        print(f"epoch: {j}")
        for i, data_ in enumerate(training_loader):
            inputs, target = data_
            target = target.view(-1, 1)
            optim.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            loss.backward()
            optim.step()
        absolute_loss = 0
        for i, data_ in enumerate(val_loader):
            inputs, target = data_
            target = target.view(-1, 1)
            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            difference = outputs - target
            absolute_loss += torch.sum(torch.abs(difference))
            print(f"loss: {loss}")
        print(f"absolute loss: {absolute_loss}")
        print(f"average absolute loss: {absolute_loss/100}")
        print("------------------------\n")

    #save model and training set
    if relu:
        base_string = "sequential_non_linear"
    else:
        base_string = "sequential_linear"
    with open(f'saved_networks/{base_string}_training_set.pkl', 'wb') as f:
        pickle.dump(training_set, f)
    save_model(net, base_string)
    return net, training_set


def get_bigger_sequential_fully_linear_trained() -> Tuple[nn.Sequential, MinDataSetFCNN]:
    return get_bigger_sequential_linear_trained(relu=False)


def get_bigger_sequential_non_linear_trained() -> Tuple[nn.Sequential, MinDataSetFCNN]:
    return get_bigger_sequential_linear_trained(relu=True)


def get_non_zero_padding_net() -> nn.Sequential:
    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, padding=1, bias=False, padding_mode="replicate"),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(9, 1, bias=False)
    )
    model[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    model[3].weight.data = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float)
    save_model(model, "non_zero_padding")
    return model


def get_convolutional_example_linear() -> nn.Sequential:
    return get_convolutional_example(relu=False)


def get_convolutional_example_non_linear() -> nn.Sequential:
    return get_convolutional_example(relu=True)


def get_convolutional_example(relu=True) -> nn.Sequential:
    """
    creates a small convolutional model.

    Whether the net has ReLU as activation function or not can be set with the relu argument. The model is saved as
    pickle and torchscript under the name super_simple_convolutional_linear or super_simple_convolutional_non_linear in
    the folder saved_networks depending on the relu parameter.

    Returns:
        the model with set weights
    """
    if relu:
        net = nn.Sequential(
            nn.Conv2d(1, 1, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(1, 1, 2, bias=False),
            nn.ReLU(),
            nn.Flatten()
        )
        net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
        net[2].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    else:
        net = nn.Sequential(
            nn.Conv2d(1, 1, 2, bias=False),
            nn.Conv2d(1, 1, 2, bias=False),
            nn.Flatten()
        )
        net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
        net[1].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    if relu:
        model_name = "super_simple_convolutional_non_linear"
    else:
        model_name = "super_simple_convolutional_linear"
    save_model(net, model_name)
    return net


def get_tiny_flatten_before_non_linear() -> nn.Sequential:
    """
    creates a small model with a flatten layer.

    The model is saved as pickle and torchscript under the name simple_flatten_before_non_linear in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, 2, bias=False),
        nn.Flatten(),
        nn.ReLU()
    )
    net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    save_model(net, "simple_flatten_before_non_linear")
    return net


def save_model(model: nn.Module, name: str):
    """
    saves a model into the folder saved_networks.

    Two copies of the model are saved, one as pickle and one as torchscript.

    Args:
        model: the model to save
        name: the path to save the model to
    """
    model_scripted = pt.jit.script(model)  # Export to TorchScript
    file_path = os.path.dirname(__file__)
    folder_path = os.path.join(file_path, "saved_networks")
    model_scripted.save(os.path.join(folder_path, f"{name}.pt"))
    with open(os.path.join(folder_path, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)


def create_tiny_avgpool() -> nn.Sequential:
    """
    creates a small model with an average pooling layer.

    also saves the model under the name tiny_avgpool in the folder saved_networks.

    Returns:
        the created model
    """
    model = nn.Sequential(
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    save_model(model, "tiny_avgpool")
    return model


def create_tiny_maxpool() -> nn.Sequential:
    """
    creates a small model with a max pooling layer.

    also saves the model under the name tiny_maxpool in the folder saved_networks.

    Returns:
        the created model
    """
    model = nn.Sequential(
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    save_model(model, "tiny_maxpool")
    return model


def create_tiny_flatten_first():
    """
    creates a small model with a flatten layer as the first layer.

    also saves the model under the name tiny_flatten_first in the folder saved_networks.

    Returns:
        the created model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2, 1, bias=False)
    )
    model[1].weight.data = torch.tensor([[0, 1]], dtype=torch.float)
    save_model(model, "tiny_flatten_first")
    return model


def create_tiny_ReLU_first():
    """
    creates a small model with a ReLU layer as the first layer.

    also saves the model under the name tiny_ReLU_first in the folder saved_networks.

    Returns:
        the created model
    """
    model = nn.Sequential(
        nn.ReLU(),
        nn.Linear(2, 1, bias=False)
    )
    model[1].weight.data = torch.tensor([[0, 1]], dtype=torch.float)
    save_model(model, "tiny_ReLU_first")
    return model


def get_convolutional_example_multi_channel():
    """
    creates a small convolutional model with multiple channels.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_multi_channel in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 2, 2, bias=False),
        nn.Conv2d(2, 1, 2, bias=False)
    )
    net[0].weight.data = torch.tensor([[[[1, 1], [0, 0]]], [[[0, 0], [1, 1]]]], dtype=torch.float)
    net[1].weight.data = torch.tensor([[[[1, 1], [0, 0]], [[1, 1], [0, 0]]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_multi_channel")
    return net


def get_convolutional_example_2d_max_pooling():
    """
    creates a small convolutional model with max pooling.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_2d_max_pooling in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, 2, bias=False),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_2d_max_pooling")
    return net


def get_convolutional_example_2d_avg_pooling():
    """
    creates a small convolutional model with average pooling.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_2d_avg_pooling in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, 2, bias=False),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_2d_avg_pooling")
    return net


def get_short_example_2d_avg_pooling():
    """
    creates a toy model that only contains 2d average pooling and relu layers.

    The model is saved as pickle and torchscript under the name super_short_2d_avg_pooling in the folder
    saved_networks.

    Returns:
        the model
    """
    net = nn.Sequential(
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten()
    )
    save_model(net, "super_short_2d_avg_pooling")
    return net


def get_convolutional_example_2d_max_pooling_multi_channel():
    """
    creates a small convolutional model with max pooling and multiple channels.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_2d_max_pooling_multi_channel
    in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 2, 2, bias=False),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(2, 1, bias=False),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]], [[[0, 0], [0, 0]]]], dtype=torch.float)
    net[6].weight.data = torch.tensor([[1, 1]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_2d_max_pooling_multi_channel")
    return net


def get_rescale_pos_neg_model():
    """
    small model with two ReLU layers in sequence.

    The model is saved as pickle and torchscript under the name rescale_pos_neg_model in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        nn.ReLU()
    )
    net[0].weight.data = torch.tensor([[1, 1, 1], [-1, 1, 0]], dtype=torch.float)
    save_model(net, "rescale_pos_neg_model")
    return net


def get_convolutional_example_2d_avg_pooling_multi_channel():
    """
    creates a small convolutional model with average pooling and multiple channels.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_2d_avg_pooling_multi_channel
    in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 2, 2, bias=False),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(2, 1, bias=False)
    )
    net[0].weight.data = torch.tensor([[[[1, 0], [0, 0]]], [[[0, 0], [0, 0]]]], dtype=torch.float)
    net[6].weight.data = torch.tensor([[1, 1]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_2d_avg_pooling_multi_channel")
    return net


def get_convolutional_example_1d_max_pooling():
    """
    creates a smalle 1-d convolutional model with max pooling.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_1d_max_pooling in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv1d(1, 1, 2, bias=False),
        nn.MaxPool1d(2),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[1, 0]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_1d_max_pooling")
    return net


def get_convolutional_example_3d_max_pooling():
    """
    creates a small 3-d convolutional model with max pooling.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_3d_max_pooling in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv3d(1, 1, 2, bias=False),
        nn.MaxPool3d(2),
        nn.ReLU(),
        nn.MaxPool3d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_3d_max_pooling")
    return net


def get_convolutional_example_non_linear_1d():
    """
    creates a small 1-d convolutional model with relu as activation function.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_non_linear_1d in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv1d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Conv1d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[1, 0]]], dtype=torch.float)
    net[2].weight.data = torch.tensor([[[1, 0]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_non_linear_1d")
    return net


def get_convolutional_example_non_linear_3d():
    """
    creates a small 3-d convolutional model with relu as activation function.

    The model is saved as pickle and torchscript under the name super_simple_convolutional_non_linear_3d in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv3d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Conv3d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Flatten()
    )
    net[0].weight.data = torch.tensor([[[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]], dtype=torch.float)
    net[2].weight.data = torch.tensor([[[[[1, 0], [0, 0]], [[0, 0], [0, 0]]]]], dtype=torch.float)
    save_model(net, "super_simple_convolutional_non_linear_3d")
    return net


def get_convolution_into_linear_mixed():
    """
    creates a small convolutional model with relu as activation function and a linear layer.

    The model is saved as pickle and torchscript under the name simple_mixed_net in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4, 1, bias=False),
        nn.ReLU()
    )
    net[0].weight.data = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float)
    net[3].weight.data = torch.tensor([[1, 0, 0, 0]], dtype=torch.float)
    save_model(net, "simple_mixed_net")
    return net


def get_convolution_into_linear_mixed_multiple_outputs():
    """
    creates a small convolutional model with relu as activation function and a linear layer with multiple outputs.

    The model is saved as pickle and torchscript under the name simple_mixed_net_multiple_outputs in the folder
    saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Conv2d(1, 1, 2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4, 2, bias=False),
        nn.ReLU()
    )
    net[0].weight.data = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float)
    net[3].weight.data = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.float)
    save_model(net, "simple_mixed_net_multiple_outputs")
    return net


def get_tiny_linear_sigmoid():
    """
    creates a two layer linear model with sigmoid as activation function.

    The model is saved as pickle and torchscript under the name tiny_linear_sigmoid in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Linear(2, 1, bias=False),
        nn.Sigmoid()
    )
    net[0].weight.data = torch.tensor([[1, 1]], dtype=torch.float)
    save_model(net, "tiny_linear_sigmoid")
    return net


def get_small_linear_sigmoid():
    """
    creates a three layer linear model with sigmoid as activation function.

    The model is saved as pickle and torchscript under the name small_linear_sigmoid in the folder saved_networks.

    Returns:
        the model with set weights
    """
    net = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.Sigmoid(),
        nn.Linear(2, 1, bias=False)
    )
    net[0].weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    net[2].weight.data = torch.tensor([[1, 1]], dtype=torch.float)
    save_model(net, "small_linear_sigmoid")
    return net


if __name__ == "__main__":
    for i in range(3):
        get_real_MNIST_net(retrain=True)
    #test()

    #get_real_MNIST_net(retrain=False, lr=0.0005, num_epochs=2, batch_size=64)

    """model = get_small_linear_sigmoid()
    test = torch.tensor([[math.log(2, math.e), math.log(2, math.e)]], dtype=torch.float)
    print(model(test))
    print(torch.sum(1 / (1 + torch.exp(-1 * test))))"""

    #save_model(nn.AvgPool2d(kernel_size=(2, 3), stride=(3, 2), padding=(1, 1)), "avg_pooling_2d")
    #get_convolution_into_linear_mixed_multiple_outputs()
    #model, training = get_bigger_sequential_linear_trained(relu=False)
    #get_sequential_linear_multiple_outputs()
    #get_sequential_non_linear_multiple_outputs()
    # get_convolutional_example_1d_max_pooling()
    # get_convolutional_example_3d_max_pooling()
    # get_convolutional_example_2d_max_pooling()
    # get_convolutional_example_2d_max_pooling_multi_channel()
    # get_convolutional_example_2d_avg_pooling()
    # get_short_example_2d_avg_pooling()
    get_convolutional_example(relu=True)

    """#create SuperMinModel and save as torchscript
    model = SuperMinModelNonLinear()
    model_scripted = pt.jit.script(model)  # Export to TorchScript
    model_scripted.save('../Deeplift_new/super_simple_example_non_linear.pt')"""

    """
    #create and train MinModel and save as torchscript
    trained_model = train_min_example()
    print(trained_model(pt.tensor([1., 1.])))
    model_scripted = pt.jit.script(trained_model)  # Export to TorchScript
    model_scripted.save('../Deeplift_new/model_scripted.pt')"""

