import importlib.util
from pathlib import Path

import torch


def load_module(name, filename):
    """Import an experiment file whose name is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


experiment = load_module("pytorch_x_squared_experiment", "02_pytorch_x_squared.py")

# Tests run on CPU so they stay fast and do not depend on a GPU being present.
DEVICE = torch.device("cpu")


def test_dataset_is_one_row_per_example():
    """The batch dimension minigrad looped over is the leading axis here."""
    x, y = experiment.make_dataset(DEVICE)

    assert x.shape == (experiment.N_SAMPLES, 1)
    assert y.shape == (experiment.N_SAMPLES, 1)
    assert torch.allclose(y, x**2)


def test_model_output_shape_matches_the_target_shape():
    model = experiment.build_model(DEVICE)
    x, y = experiment.make_dataset(DEVICE)

    with torch.no_grad():
        predictions = model(x)

    assert predictions.shape == y.shape


def test_model_has_trainable_parameters():
    model = experiment.build_model(DEVICE)
    params = list(model.parameters())

    # 1->8, 8->8, 8->1 is 16 + 72 + 9 values, the same count experiment 01 keeps
    # in its flat per-layer lists.
    assert len(params) == 6
    assert sum(p.numel() for p in params) == 97
    assert all(p.requires_grad for p in params)


def test_backward_fills_in_grads_that_the_optimizer_then_consumes():
    """The step minigrad wrote by hand is now the optimizer reading .grad."""
    model = experiment.build_model(DEVICE)
    x, y = experiment.make_dataset(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=experiment.LEARNING_RATE)

    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())

    loss = criterion(model(x), y)
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())

    before = model.layers[4].bias.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.layers[4].bias)


def test_short_training_run_reduces_the_loss():
    model = experiment.build_model(DEVICE)
    x, y = experiment.make_dataset(DEVICE)

    losses = experiment.train(model, x, y, experiment.LEARNING_RATE, 100)

    assert losses[-1] < losses[0] / 3


def test_training_is_reproducible():
    x, y = experiment.make_dataset(DEVICE)

    first = experiment.train(experiment.build_model(DEVICE), x, y, experiment.LEARNING_RATE, 50)
    second = experiment.train(experiment.build_model(DEVICE), x, y, experiment.LEARNING_RATE, 50)

    assert first == second


def test_predictions_are_close_to_targets_after_training():
    model = experiment.build_model(DEVICE)
    x, y = experiment.make_dataset(DEVICE)

    experiment.train(model, x, y, experiment.LEARNING_RATE, 400)

    with torch.no_grad():
        errors = (model(x) - y).abs().max().item()

    assert errors < 0.2


def test_agrees_with_the_minigrad_implementation():
    """Both implementations must drive the loss far down on the same curve.

    The two experiments seed their weights differently, so the exact numbers
    differ; what has to match is that both converge.
    """
    model = experiment.build_model(DEVICE)
    x, y = experiment.make_dataset(DEVICE)
    losses = experiment.train(model, x, y, experiment.LEARNING_RATE, 400)

    minigrad_experiment = load_module("x_squared_experiment", "01_x_squared.py")
    dataset = minigrad_experiment.make_dataset()
    params = minigrad_experiment.init_params()
    minigrad_experiment.train(params, dataset, minigrad_experiment.LEARNING_RATE, 400)
    minigrad_loss = minigrad_experiment.mse_loss(params, dataset).data

    assert losses[-1] < 0.05
    assert minigrad_loss < 0.05