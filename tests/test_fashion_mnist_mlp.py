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
experiment = load_module("fashion_mnist_mlp_experiment", "03_fashion_mnist_mlp.py")

DEVICE = torch.device("cpu")


def test_flatten_images_reshapes_a_grid_into_a_vector():
    images = torch.arange(2 * 1 * 28 * 28, dtype=torch.float32).view(2, 1, 28, 28)

    flat = experiment.flatten_images(images)

    assert flat.shape == (2, 784)
    assert torch.equal(flat, images.view(2, 784))


def test_model_accepts_a_batch_of_flattened_784_inputs():
    model = experiment.build_model(DEVICE)
    x = torch.randn(8, 784)

    logits = model(x)

    assert logits.shape[0] == 8


def test_output_shape_is_batch_by_ten_classes():
    model = experiment.build_model(DEVICE)
    x = torch.randn(8, 784)

    logits = model(x)

    assert logits.shape == (8, 10)


def test_outputs_are_raw_logits_usable_directly_by_cross_entropy():
    """The model must return unnormalized logits, not softmax probabilities.

    CrossEntropyLoss applies log_softmax itself, so passing raw logits works and
    produces a finite loss and gradients. A softmaxed output would double-apply
    the transform, so we also confirm the outputs are not already a probability
    distribution.
    """
    model = experiment.build_model(DEVICE)
    x = torch.randn(8, 784)
    labels = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    logits = model(x)

    assert not torch.allclose(logits.sum(dim=1), torch.ones(8), atol=1e-3)

    loss = criterion(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_model_has_trainable_parameters_and_gradients_flow_through_it():
    model = experiment.build_model(DEVICE)
    params = list(model.parameters())
    x = torch.randn(4, 784)
    labels = torch.tensor([0, 1, 2, 3])

    assert len(params) == 6
    assert sum(p.numel() for p in params) == 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
    assert all(p.requires_grad for p in params)

    nn.CrossEntropyLoss()(model(x), labels).backward()

    assert all(p.grad is not None for p in params)
    assert any(p.grad.abs().sum() > 0 for p in params)


def test_split_train_validation_is_sized_seeded_and_disjoint():
    features = torch.randn(100, 1)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(features, labels)

    train_a, val_a = experiment.split_train_validation(dataset, validation_fraction=0.2, seed=0)
    train_b, val_b = experiment.split_train_validation(dataset, validation_fraction=0.2, seed=0)

    assert len(train_a) == 80
    assert len(val_a) == 20
    assert len(train_a) + len(val_a) == len(dataset)

    # Same seed -> same split; train and validation do not overlap.
    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)
    assert set(train_a.indices).isdisjoint(val_a.indices)


def test_evaluate_reports_loss_and_accuracy_without_gradients():
    """A model rigged to predict each sample's label must score 100%."""
    n_classes = 10
    n_samples = 20

    model = nn.Linear(784, n_classes, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        for i in range(n_classes):
            model.weight[i, i] = 10.0

    images = torch.zeros(n_samples, 1, 28, 28)
    labels = torch.arange(n_samples) % n_classes
    for i in range(n_samples):
        images[i, 0, 0, labels[i]] = 1.0

    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    loss, accuracy = experiment.evaluate(model, loader, nn.CrossEntropyLoss(), DEVICE)

    assert accuracy == 1.0
    assert loss > 0
    assert all(p.grad is None for p in model.parameters())


def test_train_one_epoch_returns_a_finite_loss_and_updates_parameters():
    model = experiment.build_model(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    before = model.layers[0].weight.detach().clone()
    loss = experiment.train_one_epoch(model, loader, criterion, optimizer, DEVICE)

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0
    assert not torch.equal(before, model.layers[0].weight)
