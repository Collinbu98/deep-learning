import importlib.util
from pathlib import Path

import pytest

from minigrad.value import Value, backward


def load_experiment():
    """Import experiments/01_x_squared.py, whose name is not a valid module name."""
    path = Path(__file__).resolve().parents[1] / "experiments" / "01_x_squared.py"
    spec = importlib.util.spec_from_file_location("x_squared_experiment", path)

    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)

    return experiment


experiment = load_experiment()


def test_dataset_covers_the_target_range():
    dataset = experiment.make_dataset()

    assert len(dataset) == experiment.N_SAMPLES
    assert dataset[0][0] == -1.0
    assert dataset[-1][0] == 1.0
    assert all(x * x == target for x, target in dataset)


def test_forward_matches_a_hand_computed_layer(monkeypatch):
    """A 1 -> 1 network, so the answer is just w * x + b."""
    monkeypatch.setattr(experiment, "ARCHITECTURE", [1, 1])
    params = [[Value(3.0), Value(2.0)]]

    assert experiment.forward(5.0, params).data == pytest.approx(17.0)


def test_forward_returns_a_value_that_can_be_differentiated():
    params = experiment.init_params()

    prediction = experiment.forward(0.5, params)

    assert isinstance(prediction, Value)
    assert isinstance(prediction.data, float)


def test_predictions_are_close_to_targets_after_training():
    params = experiment.init_params()
    dataset = experiment.make_dataset()

    experiment.train(params, dataset, experiment.LEARNING_RATE, 300)

    errors = [abs(experiment.forward(x, params).data - target) for x, target in dataset]

    assert max(errors) < 0.1


def test_gradients_match_finite_differences():
    """The autograd gradients this experiment trains on must be the real ones."""
    dataset = experiment.make_dataset()
    params = experiment.init_params()

    backward(experiment.mse_loss(params, dataset))

    eps = 1e-6

    for layer in params:
        p = layer[len(layer) // 2]

        original = p.data
        p.data = original + eps
        plus = experiment.mse_loss(params, dataset).data
        p.data = original - eps
        minus = experiment.mse_loss(params, dataset).data
        p.data = original

        numeric = (plus - minus) / (2 * eps)

        assert p.grad == pytest.approx(numeric, rel=1e-5)


def test_training_reduces_the_loss():
    dataset = experiment.make_dataset()
    params = experiment.init_params()

    before = experiment.mse_loss(params, dataset).data
    losses = experiment.train(params, dataset, experiment.LEARNING_RATE, 300)

    assert losses[0] == pytest.approx(before)
    assert losses[-1] < losses[0] / 100


def test_excessive_learning_rate_does_not_reduce_the_loss():
    dataset = experiment.make_dataset()
    params = experiment.init_params()

    losses = experiment.train(params, dataset, experiment.TOO_BIG_LEARNING_RATE, 100)

    assert losses[-1] > losses[0]


def test_training_is_reproducible():
    dataset = experiment.make_dataset()

    first = experiment.train(experiment.init_params(), dataset, 0.1, 50)
    second = experiment.train(experiment.init_params(), dataset, 0.1, 50)

    assert first == second
