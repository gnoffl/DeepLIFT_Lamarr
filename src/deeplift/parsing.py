from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Iterable, Generator, Callable
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

    @abstractmethod
    def _add_layer(self, layer: Callable, layer_name: str):
        """
        adds a layer to the model

        Args:
            layer: the layer to be added. Just needs to be callable.
            layer_name: name of the layer.
        """
        pass

    def _remove_layer(self, layer_name: str):
        """
        removes a layer from the model

        Args:
            layer_name: name of the layer to be removed
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


class GraphNode(ABC):
    """
    abstract class for nodes in the computation graph

    !!!not in use at the moment, also not functional!!!
    """
    debug_name: str
    type: str

    def __str__(self) -> str:
        """
        returns string of the node
        :return: string of the node
        """
        return f"{self.debug_name} ({self.type})"

    def __repr__(self) -> str:
        """
        returns string representation of the node
        :return: string representation of the node
        """
        return self.__str__()


class GraphValueObject(GraphNode):
    """
    class for objects in the computation graph that contain values like tensors
    """
    output_from: "GraphCallableNode"
    input_for: List[GraphNode]

    def __init__(self, debug_name: str, node_type: str, output_from: "GraphCallableNode" = None,
                 input_for: List[GraphNode] = None) -> None:
        """
        initializes the node
        :param debug_name: debug name used in the jit description
        :param node_type: type of the node
        :param output_from: GraphCallableNode that create this object as output
        :param input_for: GraphCallableNodes that use this object as input
        """
        self.debug_name = debug_name
        self.type = node_type
        self.output_from = output_from
        self.input_for = input_for if input_for is not None else []


class GraphCallableNode(GraphNode):
    """
    class for callable nodes in the computation graph
    """
    input: List[GraphValueObject]
    output: List[GraphValueObject]

    def __init__(self, debug_name: str, node_type: str, input_values: List[GraphNode] = None,
                 output: List[GraphNode] = None) -> None:
        """
        initializes the node
        :param debug_name: debug name used in the jit description
        :param node_type: type of the node
        :param input_values: value objects that are used as input for the node
        :param output: value objects that are created as output of the node
        """
        self.debug_name = debug_name
        self.type = node_type
        self.input = [] if input_values is None else input_values
        self.output = [] if output is None else output

    def add_input(self, input_node: GraphValueObject) -> None:
        """
        adds a value object as input for the node
        :param input_node: value object that is used as input
        """
        self.input.append(input_node)

    def add_output(self, output_node: GraphValueObject) -> None:
        """
        adds a value object as output for the node
        :param output_node: value object that is created as output
        """
        self.output.append(output_node)


class ComplexLoadedModel(nn.Module, LoadedModel):
    """
    class for loaded models that are parsed from a jit graph representation using maximum generalization
    """
    nodes: List[GraphNode]
    layers: OrderedDict

    def __init__(self, model: torch.jit._script.RecursiveScriptModule, test=False) -> None:
        """
        initializes the model by especially parsing the layers and creating a dict with the relevant information
        :param model: base model
        """
        super(ComplexLoadedModel, self).__init__()
        self.layers = OrderedDict()
        self.model = model
        if test:
            self.nodes = self.parse_model_compute_graph_to_graph_nodes(model)
        else:
            self.parse_model_compute_graph(model)
        #self.parse_model_torchscript(model)

    def parse_model_torchscript(self, model: torch.jit._script.RecursiveScriptModule) -> None:
        """
        parses the model by creating a dictionary with the layer names and types using the torchscript description
        :param model: model which is to be examined
        """
        model_description = str(model)
        lines: List[str] = model_description.split("\n")
        lines = [line.strip() for line in lines]
        for line in lines:
            self.parse_model_line_torchscript(line)
        for val in self.layers.values():
            if val not in implemented_layer_types:
                raise ParsingError(f"unknown layer type {val}!")

    def parse_model_compute_graph(self, model: torch.jit._script.RecursiveScriptModule) -> None:
        """
        parses the model by creating a dictionary with the layer names and types using the computation graph
        :param model: model which is to be examined
        """
        graph = model.graph
        #get layer names and types from graph
        for node in graph.nodes():
            print_node(node)
            #parse layers
            if node.kind() == "prim::GetAttr":
                if node.output().type().kind() == "ClassType":
                    layer_type = node.output().type().name()
                    layer_name = node.s("name")
                    if layer_name in self.layers.keys():
                        raise ParsingError(f"name {layer_name} already exists in model!")
                    self.layers[layer_name] = layer_type
            #parse functions
            elif node.kind() == "prim::CallFunction":
                if node.inputsAt(0).type().kind() == "ClassType":
                    layer_type = node.inputsAt(0).type().name()
                    layer_name = node.inputsAt(0).debugName()
                    if layer_name in self.layers.keys():
                        raise ParsingError(f"name {layer_name} already exists in model!")
                    self.layers[layer_name] = layer_type

    def append_node(self, node: GraphNode) -> None:
        if node.debug_name in [graph_nodes.debug_name for graph_nodes in self.nodes]:
            raise ParsingError(f"name {node.debug_name} already exists in model!")
        self.nodes.append(node)

    def get_node(self, debug_name: str) -> GraphNode:
        for node in self.nodes:
            if node.debug_name == debug_name:
                return node
        raise ValueError(f"node with name {debug_name} does not exist!")

    #print nodes in model in ordered fashion
    def print_nodes(self) -> None:
        print("[", end="")
        for i, node in enumerate(self.nodes):
            print(node, end="")
            if i < len(self.nodes) - 1:
                print(", ", end="")
        print("]")

    #parse model from computation graph to a List of GraphNode Objects
    def parse_model_compute_graph_to_graph_nodes(self, model: torch.jit._script.RecursiveScriptModule) -> None:
        """
        parses the model by creating a dictionary with the layer names and types using the computation graph
        :param model: model which is to be examined
        """
            #parse inputs
            #parse missing node types
        graph = model.graph
        print(graph)
        if (not hasattr(self, "nodes")) or self.nodes is None:
            self.nodes = []
        else:
            raise ParsingError("nodes already exist!")

        #get inputs from graph
        for input_node in graph.inputs():
            self.nodes.append(GraphValueObject(debug_name=input_node.debugName(), node_type=str(input_node.type())))

        fails = []
        # get layer names and types from graph
        for node in graph.nodes():
            print_node(node)
            # parse layers
            if node.kind() == "prim::GetAttr":
                self.parse_getattr_to_node(fails, node)
            # parse functions
            elif node.kind() == "prim::CallFunction":
                self.parse_callfunction_to_node(fails, node)
            elif node.kind() == "prim::Constant":
                self.parse_constant_to_node(fails=fails, node=node)
            elif node.kind() == "prim::CallMethod":
                self.parse_callmethod_to_node(fails=fails, node=node)
            else:
                fails.append(f"node.kind == {node.kind()}")

        self.print_nodes()
        print(fails)

    def parse_callmethod_to_node(self, fails: List[str], node: torch._C.Node) -> None:
        #create GraphValueObject
        name = node.output().debugName()
        val_type = node.output().type()
        val_obj = GraphValueObject(debug_name=name, node_type=str(val_type))
        #add GraphValueObject to graph
        self.append_node(val_obj)
        #add GraphValueObject to output for called node
        self.create_inputs_and_outputs(node=node, fails=fails, val_obj=val_obj)

    def create_inputs_and_outputs(self, node, fails, val_obj) -> None:
        called_thing_node: GraphNode = None
        for i, node_name in enumerate(node.inputs()):
            if i == 0:
                #this is the output of the function/Layer
                try:
                    called_thing_node = self.get_node(debug_name=node_name.debugName())
                    assert isinstance(called_thing_node, GraphCallableNode)
                    called_thing_node.add_output(val_obj)
                except ValueError as e:
                    fails.append(f"Function/Layer \"{node_name}\" not found in graph!")
                    raise ParsingError(f"initial node \"{node_name}\" not found in graph! Problem occurred when parsing "
                                       f"{node}.")
            else:
                #these are the inputs of the function/Layer
                try:
                    input_node_object = self.get_node(debug_name=node_name.debugName())
                    assert isinstance(input_node_object, GraphValueObject)
                    called_thing_node.add_input(input_node_object)
                except ValueError as e:
                    fails.append(
                        f"input node {node_name.debugName()} not found in graph! Problem occurred when parsing "
                        f"node {node}.")

    def parse_constant_to_node(self, fails: List[str], node: torch._C.Node) -> None:
        #find out whether node is a function
        if node.output().type().kind() == "FunctionType":
            #create function node
            function_node = GraphCallableNode(debug_name=node.output().debugName(), node_type=str(node.output().type()))
            #add function node to graph
            self.append_node(function_node)
            #get input nodes
            for input_node in node.inputs():
                #find input node in graph
                for graph_node in self.nodes:
                    if graph_node.debug_name == input_node.debugName():
                        #add input node to function node
                        function_node.add_input(graph_node)
                        break
                else:
                    fails.append(f"input node {input_node.debugName()} not found in graph!")
        elif str(node.output().type()) == "bool":
            constant_node = GraphValueObject(debug_name=node.output().debugName(), node_type=str(node.output().type()))
            self.nodes.append(constant_node)
        else:
            fails.append(f"node.kind() == prim::Constant, node.output().kind() == {node.output().kind()}")

    def parse_callfunction_to_node(self, fails, node) -> None:
        if node.inputsAt(0).type().kind() == "FunctionType":
            #create GraphValueObject
            value_name = node.output().debugName()
            value_type = str(node.output().type())
            value_object = GraphValueObject(debug_name=value_name, node_type=value_type)
            #add GraphValueObject to graph
            self.append_node(value_object)
            #get function node
            self.create_inputs_and_outputs(node=node, fails=fails, val_obj=value_object)
        else:
            fails.append(
                f"node.kind() == prim::CallFunction, node.inputsAt(0).type().kind() == {node.inputsAt(0).type().kind()}")

    def parse_getattr_to_node(self, fails, node) -> None:
        if node.output().type().kind() == "ClassType":
            layer_type = node.output().type().name()
            layer_name = node.output().debugName()
            node_object = GraphCallableNode(layer_name, layer_type, [], [])
            node_object.name = node.s("name")
            self.append_node(node_object)
        else:
            fails.append(f"node.kind == ClassType, node.output.kind == {node.output().type().kind()}")

    def parse_model_line_torchscript(self, line: str) -> None:
        """
        parses a line from the description of the model to get the layers and types
        :param line: line of the description
        """
        if ": " in line:
            name, line = line.split(": ")
            name = name.strip("()")

            #check format is expected
            if name in self.layers.keys():
                raise ParsingError(f"name {name} already exists in model!")
            if not line.startswith("RecursiveScriptModule(original_name="):
                raise ParsingError(f"unexpected format of layer {name}!")
            if not line.count("=") == 1:
                raise ParsingError(f"unexpected format of layer {name}!")

            layer_type = line.split("=")[1].strip(")")
            self.layers[name] = layer_type


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
            searched_layer_name: the name of the layer

        Returns:
            the layer object
        """
        for layer_name, layer in self.layers:
            if searched_layer_name == layer_name:
                return layer
        raise ValueError(f"layer \"{searched_layer_name}\" not found in model!")

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
        layers = ["input"]
        layers.extend([layer_name for layer_name, layer in self.layers])
        return layers

    def _add_layer(self, layer: Callable, layer_name: str):
        """
        adds a layer to the model

        Args:
            layer: the layer to be added. Just needs to be callable.
            layer_name: Name of the layer.
        """
        self.layers.append((layer_name, layer))

    def _remove_layer(self, layer_name: str):
        """
        removes a layer from the model

        Args:
            layer_name: name of the layer to be removed
        """
        for i, (name, layer) in enumerate(self.layers):
            if layer_name == name:
                self.layers.pop(i)
                break
        else:
            raise ValueError(f"layer {layer_name} not found in model!")


def print_node(node) -> None:
    """
    prints information about a node
    :param node: the node to be printed
    """
    print(node)
    print(f"node.kind: {node.kind()}")
    if node.kind() == "prim::GetAttr":
        print_getattr(node)
        if node.output().type().kind() == "ClassType":
            print_getattr_classtype(node)
    elif node.kind() == "prim::CallFunction" and node.inputsAt(0).type().kind() == "ClassType":
        print_callfunction_classtype(node)
    elif node.kind() == "prim::Constant":
        print_constant(node)
    print("--------------------------")


def print_constant(node) -> None:
    """
    prints information about a constant node
    :param node: node to be printed
    """
    print(f"node.output: {node.output()}")
    print(f"node.output.debugname: {node.output().debugName()}")
    print(f"node.output.type: {node.output().type()}")
    if hasattr(node.output().type(), "name"):
        print(f"node.output.type.name: {node.output().type().name()}")
    else:
        print("node.output.type.name: None")


def print_getattr_classtype(node) -> None:
    """
    prints information about a getattr classtype node
    :param node: node to be printed
    """
    print(f"node.output.type.name: {node.output().type().name()}")
    print(f"node.s(\"name\"): {node.s('name')}")


def print_getattr(node) -> None:
    """
    prints information about a general getattr node
    :param node: node to be printed
    """
    print(f"node.output: {node.output()}")
    print(f"node.output.type: {node.output().type()}")
    print(f"node.output.type.kind: {node.output().type().kind()}")


def print_callfunction_classtype(node) -> None:
    """
    prints information about a callfunction classtype node
    :param node: node to be printed
    """
    print(f"node.inputsAt(0): {node.inputsAt(0)}")
    print(f"node.inputsAt(0).type: {node.inputsAt(0).type()}")
    print(f"node.inputsAt(0).type.kind: {node.inputsAt(0).type().kind()}")
    print(f"node.inputsAt(0).type.name: {node.inputsAt(0).type().name()}")
    print(f"node.inputsAt(0).debugName: {node.inputsAt(0).debugName()}")


def load_net_complex(model_path: str, test: bool = False) -> ComplexLoadedModel:
    """
    loads a model using the ComplexLoadedModel class
    :param model_path: path to the saved model file
    :param test: determines in which way the model is loaded. If True, the model is loaded in test mode, which means
    using the whole range of the computation graph and parsing to nodes.
    :return: the loaded model
    """
    net = torch.jit.load(model_path)
    net.eval()
    loaded = ComplexLoadedModel(net, test)
    print(loaded.layers)
    return loaded


def load_net_sequential(model_path: str) -> SequentialLoadedModel:
    """
    loads a model using the SequentialLoadedModel class
    :param model_path: path to the saved model file
    :return: the loaded model
    """
    net = torch.jit.load(model_path)
    net = SequentialLoadedModel(net)
    return net


if __name__ == "__main__":
    #test_rules(method="linear", model="super_simple_example.pt")
    #load_net_complex(model="super_simple_example_non_linear.pt", test=True)
    loaded = load_net_sequential(model_path='Tests/saved_networks/super_simple_sequential.pt')
    #get weights for first layer
    print(loaded.layers[0][1].weight)

