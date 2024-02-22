# DeepLIFT Lamarr

This is a reimplementation of the DeepLIFT algorithm in PyTorch which aims to attribute contributions
to the difference of an output to a reference output to the input values. The original DeepLIFT paper
can be found [here](https://arxiv.org/pdf/1704.02685.pdf) and its improved version to approximate
Shapley values can be found [here](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).

## Usage
```python
import torch
import torch.nn as nn
import Deeplift.Deeplift_new.deeplift as deeplift
import Deeplift.Deeplift_new.parsing as parsing

my_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

explainer = deeplift.DeepLiftClass(model=parsing.SequentialLoadedModel(torch.jit.script(my_model)),
                                   reference_value=torch.zeros(1, 10))
explanations, _ = explainer.attribute(torch.ones(1, 10))
print(explanations)
```

## Current restrictions

### Models
Models for which contributions are calculated have to be given in the torchscript format. Models
have to be of type nn.Sequential and activation functions must be given as their own layer instead of
them being part of other layers. The output of models needs to be one dimensional. Explanations can
also be calculated with respect to an error function. The only error function currently supported is
MSE. Only certain layers are supported. These are:
- Linear
- Conv1d
- Conv2d
- Conv3d
- MaxPool1d
- MaxPool2d
- MaxPool3d
- AvgPool1d
- AvgPool2d
- AvgPool3d
- Flatten
- ReLU
- Sigmoid
- Dropout

### Input
Input values always need to contain a batch dimension. The batch dimension is always the first
dimension. The input values need to be of type torch.Tensor.
