import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_module(name, filename):
    """Import an experiment file whose name is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# These tests never download Fashion-MNIST and never run full training. They use
# small synthetic tensors so they are fast and work offline.
experiment = load_module("fashion_mnist_cnn_experiment", "04_fashion_mnist_cnn.py")

DEVICE = torch.device("cpu")


def _batch_and_labels(n):
    return torch.randn(n, 1, 28, 28), torch.randint(0, 10, (n,))


def test_conv_input_shape_produces_batch_by_ten_output():
    model = experiment.build_model(DEVICE)
    images, _ = _batch_and_labels(4)

    logits = model(images)

    assert logits.shape == (4, 10)


def test_model_has_trainable_parameters_and_gradients_flow_through_both_stacks():
    model = experiment.build_model(DEVICE)
    params = list(model.parameters())

    assert sum(p.numel() for p in params) == 206922
    assert all(p.requires_grad for p in params)

    images, labels = _batch_and_labels(4)
    nn.CrossEntropyLoss()(model(images), labels).backward()

    # Both the convolutional stack and the classifier must receive gradients.
    assert model.features[0].weight.grad is not None
    assert model.classifier[1].weight.grad is not None
    assert model.features[0].weight.grad.abs().sum() > 0
    assert model.classifier[3].weight.grad.abs().sum() > 0


def test_outputs_are_raw_logits_usable_directly_by_cross_entropy():
    model = experiment.build_model(DEVICE)
    images, labels = _batch_and_labels(8)
    criterion = nn.CrossEntropyLoss()

    logits = model(images)

    # Raw logits are not a probability distribution (a softmaxed model would
    # already sum to 1 per row and double-apply the loss's internal log_softmax).
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8), atol=1e-3)

    loss = criterion(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_spatial_dimensions_survive_the_conv_and_pool_layers():
    """28x28 -> conv -> 28x28 -> pool -> 14x14 -> conv -> 14x14 -> pool -> 7x7."""
    model = experiment.build_model(DEVICE)
    images, _ = _batch_and_labels(2)

    with torch.no_grad():
        after_conv1 = model.features[:2](images)  # conv + relu
        after_pool1 = model.features[:3](images)  # + maxpool
        after_conv2 = model.features[:5](images)  # + conv + relu
        feature_maps = model.forward_features(images)  # + final maxpool

    assert after_conv1.shape == (2, 16, 28, 28)
    assert after_pool1.shape == (2, 16, 14, 14)
    assert after_conv2.shape == (2, 32, 14, 14)
    assert feature_maps.shape == (2, 32, 7, 7)


def test_a_small_synthetic_batch_completes_a_training_step():
    model = experiment.build_model(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    images, labels = _batch_and_labels(16)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    before = model.features[0].weight.detach().clone()
    loss = experiment.train_one_epoch(model, loader, criterion, optimizer, DEVICE)

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0
    assert not torch.equal(before, model.features[0].weight)
