"""
Experiment 01 - fitting y = x^2 with a 1 -> 8 -> 8 -> 1 MLP.

What we are learning
    A tiny network that maps a single scalar x to x squared.

Why x^2 is a good first task
    x^2 is not linear. A network with no nonlinearity in its hidden layers can
    only ever draw straight lines, so it cannot fit x^2 no matter how long it
    trains. This makes the task a real test of the whole pipeline: if the tanh
    activations, the backward pass, or the parameter updates are wrong, the
    loss plateaus and stays high. The answer is also known exactly, so every
    prediction can be checked against the truth by eye.

What this experiment demonstrates
    The full training loop written out by hand: forward pass, MSE loss,
    zero_grad, backward, parameter update. Gradients come from the tiny scalar
    autograd in src/minigrad, one Python float at a time - no PyTorch, no
    NumPy. A second run with a deliberately excessive learning rate shows what
    "too fast" looks like: each step overshoots so badly that the loss grows
    without bound and the weights blow up to inf, then to nan.

Run with: .venv/bin/python experiments/01_x_squared.py
"""

import random

from minigrad.value import Value, backward

SEED = 0
ARCHITECTURE = [1, 8, 8, 1]
N_SAMPLES = 21
LEARNING_RATE = 0.15
TOO_BIG_LEARNING_RATE = 0.25
STEPS = 400
LOG_EVERY = 40


def make_dataset():
    """N_SAMPLES evenly spaced x values over [-1, 1], with targets y = x^2."""
    return [
        (x, x * x)
        for x in [-1.0 + 2.0 * i / (N_SAMPLES - 1) for i in range(N_SAMPLES)]
    ]


def init_params(seed=SEED):
    """Random weights in [-1, 1] and biases in [-1, 1], one flat list per layer."""
    random.seed(seed)

    def rand():
        return Value(2.0 * random.random() - 1.0)

    return [
        [rand() for _ in range(n_in * n_out + n_out)]
        for n_in, n_out in zip(ARCHITECTURE, ARCHITECTURE[1:])
    ]


def forward(x, params):
    """Push one scalar x through the layers, applying tanh between them.

    Each layer is stored flat as [weights..., biases...], with the weights laid
    out row-major as n_out rows of n_in values. So one neuron is just: take the
    row of weights for that neuron, multiply by the incoming activations, add
    the bias. tanh goes between layers but not after the last one, which must
    stay linear for the output to be able to reach any value.
    """
    activations = [Value(x)]

    for layer_index, (layer, n_in, n_out) in enumerate(
        zip(params, ARCHITECTURE, ARCHITECTURE[1:])
    ):
        weights = layer[: n_in * n_out]
        biases = layer[n_in * n_out :]

        next_activations = []
        for neuron in range(n_out):
            row = weights[neuron * n_in : (neuron + 1) * n_in]
            acc = biases[neuron]
            for j, w in enumerate(row):
                acc = acc + activations[j] * w
            next_activations.append(acc)

        if layer_index < len(params) - 1:
            next_activations = [a.tanh() for a in next_activations]

        activations = next_activations

    return activations[0]


def mse_loss(params, dataset):
    """Mean squared error between the network's predictions and the targets."""
    total = Value(0.0)
    for x, target in dataset:
        error = forward(x, params) - target
        total = total + error * error

    return total / len(dataset)


def train(params, dataset, learning_rate, steps):
    """Plain gradient descent: forward, loss, zero grads, backward, update."""
    losses = []

    for _ in range(steps):
        loss = mse_loss(params, dataset)

        for layer in params:
            for p in layer:
                p.zero_grad()

        backward(loss)

        for layer in params:
            for p in layer:
                p.data -= learning_rate * p.grad

        losses.append(loss.data)

    return losses


def show_predictions(params, dataset, title):
    print()
    print(title)
    print(f"  {'x':>6}  {'target':>7}  {'predicted':>9}  {'error':>7}")
    for x, target in dataset:
        pred = forward(x, params).data
        print(f"  {x:6.2f}  {target:7.4f}  {pred:9.4f}  {pred - target:7.4f}")


def show_progress(losses, learning_rate):
    """Print the loss every LOG_EVERY steps, always including the last one."""
    print(f"training for {len(losses)} steps at learning rate {learning_rate}")

    logged = range(0, len(losses), LOG_EVERY)
    for step in logged:
        print(f"  step {step:4d}  loss {losses[step]:.6g}")
    if len(losses) - 1 not in logged:
        print(f"  step {len(losses) - 1:4d}  loss {losses[-1]:.6g}")


def main():
    dataset = make_dataset()
    params = init_params()

    print("Fitting y = x^2 with a 1 -> 8 -> 8 -> 1 MLP")
    print(f"{N_SAMPLES} samples from x in [-1, 1], {sum(len(p) for p in params)} parameters")

    show_predictions(params, dataset, "before training:")

    print()
    losses = train(params, dataset, LEARNING_RATE, STEPS)
    show_progress(losses, LEARNING_RATE)

    show_predictions(params, dataset, "after training:")

    print()
    print("now the same network and data again, but stepping much too far each time:")
    diverging_losses = train(init_params(), dataset, TOO_BIG_LEARNING_RATE, STEPS)
    show_progress(diverging_losses, TOO_BIG_LEARNING_RATE)

    print()
    print(f"learning rate {LEARNING_RATE}: {losses[0]:.6g} -> {losses[-1]:.6g}  (converging)")
    print(
        f"learning rate {TOO_BIG_LEARNING_RATE}: {diverging_losses[0]:.6g}"
        f" -> {diverging_losses[-1]:.6g}  (diverging)"
    )


if __name__ == "__main__":
    main()

