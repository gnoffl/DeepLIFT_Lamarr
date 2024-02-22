import torch
import torch.nn as nn
from deeplift import deeplift
from deeplift import parsing


def test():
    my_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    explainer = deeplift.DeepLiftClass(model=parsing.SequentialLoadedModel(torch.jit.script(my_model)),
                                       reference_value=torch.zeros(1, 10))
    explanations, _ = explainer.attribute(torch.ones(1, 10))
    print(explanations)


if __name__ == "__main__":
    test()
