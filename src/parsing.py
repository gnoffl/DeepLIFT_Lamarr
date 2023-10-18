from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Iterable, Generator
from collections import OrderedDict


implemented_layer_types = ["Linear", "Dropout"]


class ParsingError(Exception):
    pass


class LoadedModel(ABC):
    """
    Represents a pytorch model loaded from torchscript.

    Attributes:
        model: the loaded pytorch model
    """
    model: torch.jit._script.RecursiveScriptModule

    @abstractmethod
    def __len__(self) -> int:
        """
        gives the number of layers in the model.

        Returns:
            the number of layers in the model
        """
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[str, torch.jit._script.RecursiveScriptModule]:
        """
        returns layer and layer name of the layer at the given index of the model

        Args:
            item: the index of the layer of the model

        Returns:
            (layer_name, layer) where layer_name is the name of the layer, and layer is the actual layer object
        """
        pass

    @abstractmethod
    def get_layer(self, layer_name: str) -> torch.jit._script.RecursiveScriptModule:
        """
        gets a layer by name

        Args:
            layer_name: the name of the layer

        Returns:
            the layer object
        """
        pass

    #get iterator for model
    def __iter__(self) -> "LoadedModelIterator":
        """
        method for iterating over the model.

        iterates over the layers of the model in the order in which the layers are used.

        Returns:
            An iterator over the model.
        """
        return LoadedModelIterator(self)

    @abstractmethod
    def __reversed__(self) -> Iterable:
        """method for iterating over the model in a reversed manner.

        Returns:
            layers in the reversed order of their use in the model
        """
        pass

    @abstractmethod
    def get_previous_layer_name(self, layer_name: str) -> str:
        """gives the name of the previous layer

        Args:
            layer_name: name of the current layer. The name of the predecessor to this layer will be returned.

        Returns:
            name of the previous layer.
        """
        pass

    @abstractmethod
    def get_all_layer_names(self) -> List[str]:
        """gets a List of the names of all layers in the model

        Returns:
            List of the names of all layers in the model
        """
        pass


class LoadedModelIterator:
    """
    iterator for loaded models

    Attributes:
        model: the loaded model to iterate over
        index: the index of the next layer of the model to be returned
    """

    model: LoadedModel
    index: int

    def __init__(self, model: LoadedModel) -> None:
        """
        initializes the iterator object

        Args:
            model: a loaded model, that is iterated over.
        """
        self.model = model
        self.index = 0

    def __next__(self) -> Tuple[str, torch.jit._script.RecursiveScriptModule]:
        """
        gets the next layer or raises a StopIteration if there are no more layers to be returned.

        Returns:
            (layer_name, layer) where layer is the next layer of the model and layer_name the corresponding name.
        """
        if self.index < len(self.model):
            result = self.model[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


class SequentialLoadedModel(nn.Module, LoadedModel):
    """
    class for a loading sequential models

    Attributes:
        model: the Torchscript model
        layers: a list of the layers in the model in the order in which they are used in the sequential model. For each
        layer a tuple (layer_name, layer) is in the list, where layer is the layer object and layer_name the
        corresponding name
    """

    model: torch.jit._script.RecursiveScriptModule
    layers: List[Tuple[str, torch.jit._script.RecursiveScriptModule]]

    def __init__(self, model: torch.jit._script.RecursiveScriptModule = None, path: str = "") -> None:
        """
        initializes the model by using the special structure of sequential models

        Args:
            model: the sequential torchscript model
            path: path to the location of the saved sequential torchscript model
        """
        super(SequentialLoadedModel, self).__init__()

        if model is None and path == "":
            raise ValueError("either model or path must be given!")
        elif model is not None and path != "":
            raise ValueError("either model or path must be given!")

        if path != "":
            model = torch.jit.load(path)

        self.model = model
        self.layers: List[Tuple[str, torch.jit._script.RecursiveScriptModule]] = []

        # get layers and types
        for i, (layer_name, layer) in enumerate(model.named_modules()):
            if i == 0:
                continue
            self.layers.append((layer_name, layer))

    def __len__(self) -> int:
        """
        gives the number of layers in the model.

        Returns:
            the number of layers in the model
        """
        return len(self.layers)

    def __getitem__(self, item) -> Tuple[str, torch.jit._script.RecursiveScriptModule]:
        """
        returns layer and layer name of the layer at the given index of the model

        Args:
            item: the index of the layer of the model

        Returns:
            (layer_name, layer) where layer_name is the name of the layer, and layer is the actual layer object
        """
        return self.layers[item]

    def get_layer(self, searched_layer_name: str) -> torch.jit._script.RecursiveScriptModule:
        """
        gets a layer by name

        Args:
            layer_name: the name of the layer

        Returns:
            the layer object
        """
        for layer_name, layer in self.layers:
            if searched_layer_name == layer_name:
                return layer
        raise ValueError(f"layer {searched_layer_name} not found in model!")

    #reversed function to return the layers in reversed order
    def __reversed__(self) -> Generator[Tuple[str, torch.jit._script.RecursiveScriptModule], None, None]:
        """method for iterating over the model in a reversed manner.

        Returns:
            layers in the reversed order of their use in the model
        """
        for i in range(len(self.layers) - 1, -1, -1):
            yield self.layers[i]

    def get_previous_layer_name(self, layer_name: str) -> str:
        """gives the name of the previous layer

        Args:
            layer_name: name of the current layer. The name of the predecessor to this layer will be returned.

        Returns:
            name of the previous layer.
        """
        if layer_name == "input":
            return "input"
        for i, (name, layer) in enumerate(self.layers):
            if layer_name == name:
                if i == 0:
                    return "input"
                return self.layers[i - 1][0]
        raise ValueError(f"layer {layer_name} not found in model!")

    def get_all_layer_names(self) -> List[str]:
        """gets a List of the names of all layers in the model

        Returns:
            List of the names of all layers in the model
        """
        layers = [layer_name for layer_name, layer in self.layers]
        layers.append("input")
        return layers


def load_net_sequential(model_path: str) -> SequentialLoadedModel:
    """
    loads a model using the SequentialLoadedModel class
    :param model_path: path to the saved model file in torchscript format
    :return: the loaded model
    """
    net = torch.jit.load(model_path)
    net = SequentialLoadedModel(net)
    return net

