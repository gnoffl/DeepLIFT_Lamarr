from time import sleep

import tests.regression_models as regression_models
import tests.models as models
import tests.comparison as comparison
import matplotlib.pyplot as plt
import torch


def test_visualize_explanation(non_linearity="reveal_cancel"):
    test_loader, train_loader = regression_models.get_MNIST_dataloaders()
    baseline = comparison.get_MNIST_avgs(test_loader)
    new_dl = models.get_explainer("MNIST_net_no_max", baseline=baseline)
    images, _ = train_loader.__iter__().__next__()
    image = images[0:1]
    plt.imshow(image.reshape(28, 28), cmap="nipy_spectral")
    plt.show()
    result_reveal_cancel, deltas = new_dl.attribute(image, non_linearity_method="reveal_cancel")
    result_reveal_cancel = result_reveal_cancel.detach()
    result_rescale, deltas_rescale = new_dl.attribute(image, non_linearity_method="rescale")
    result_rescale = result_rescale.detach()
    min_res = torch.min(result_rescale)
    max_res = torch.max(result_rescale)
    min_val = min(torch.min(result_reveal_cancel), min_res)
    max_val = max(torch.max(result_reveal_cancel), max_res)
    diff = result_rescale - result_reveal_cancel
    for i in range(result_reveal_cancel.shape[1]):
        plt.clf()
        plt.imshow(result_reveal_cancel[0, i].reshape(28, 28), cmap="nipy_spectral", vmin=min_val, vmax=max_val)
        plt.show()
        plt.clf()
        plt.imshow(result_rescale[0, i].reshape(28, 28), cmap="nipy_spectral", vmin=min_val, vmax=max_val)
        plt.show()
        plt.clf()
        plt.imshow(diff[0, i].reshape(28, 28), cmap="nipy_spectral", vmin=min_val, vmax=max_val)
        plt.show()
        sleep(0.5)


if __name__ == "__main__":
    test_visualize_explanation("rescale")