"""
Experiment 02 - the same y = x^2 MLP, this time in PyTorch.

This is a bridge, not a new experiment. It repeats experiment 01 with the same
dataset, the same 1 -> 8 -> 8 -> 1 architecture, the same tanh activations, the
same MSE loss and the same learning rate, but written the idiomatic PyTorch
way. Putting the two side by side is the whole point: everything minigrad made
us write by hand, PyTorch does for us, and the only thing left for us to write
is the five lines of the training loop.

The correspondence
    minigrad Value                     <->  torch.Tensor
    graph built by operator overloading <->  the autograd graph
    backward(root)                     <->  loss.backward()
    grads stored on each Value         <->  tensor.grad
    p.data -= lr * p.grad              <->  optimizer.step()
    p.zero_grad() for every p          <->  optimizer.zero_grad()
    flat list of Value per layer       <->  nn.Linear's .weight / .bias

The one structural difference worth noticing: experiment 01 loops over the
dataset in Python and builds a fresh scalar graph for each x, then sums the
graphs by hand. Here the whole dataset goes through the model at once as a
single [N, 1] tensor, and PyTorch builds exactly one graph for the whole batch.
The math is identical - the sum over examples is the same sum - but the work
that minigrad did in a Python loop happens inside one batched matmul here.

Run with: .venv/bin/python experiments/02_pytorch_x_squared.py
"""

import torch
from torch import nn

SEED = 0
ARCHITECTURE = [1, 8, 8, 1]
N_SAMPLES = 21
LEARNING_RATE = 0.15
STEPS = 400
LOG_EVERY = 40


def get_device():
    """Prefer the GPU, but fall back to CPU so the script runs anywhere.

    On AMD, PyTorch reuses the name 'cuda' for ROCm, so torch.cuda.is_available()
    is the check for either vendor. This experiment is far too small to care
    about the difference; we only print it so the choice is visible.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataset(device):
    """N_SAMPLES evenly spaced x over [-1, 1] with targets y = x^2, as tensors.

    Shape is [N, 1]: one column per example, one row per feature. minigrad held
    each example as a bare Python float and looped over them; here the whole
    dataset is a single tensor and the leading dimension is the batch.
    """
    x = torch.linspace(-1.0, 1.0, N_SAMPLES, device=device).unsqueeze(1)
    y = x**2

    return x, y


class MLP(nn.Module):
    """1 -> 8 -> 8 -> 1 with tanh between layers.

    Experiment 01 stored each layer as one flat list of Value: weights first
    row-major, then biases. nn.Linear keeps the same values but splits them into
    two named tensors with the conventional [out_features, in_features] layout,
    and marks them requires_grad so autograd fills in .grad for us.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ARCHITECTURE[0], ARCHITECTURE[1]),
            nn.Tanh(),
            nn.Linear(ARCHITECTURE[1], ARCHITECTURE[2]),
            nn.Tanh(),
            nn.Linear(ARCHITECTURE[2], ARCHITECTURE[3]),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(device, seed=SEED):
    """Seed first, then construct, so the initial weights are reproducible."""
    torch.manual_seed(seed)
    model = MLP().to(device)

    return model


def train(model, x, y, learning_rate, steps):
    """Full-batch gradient descent, written out.

    The four calls below are the entire training loop. In minigrad each of them
    was a loop over parameters or a hand-written topological sort.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []

    for _ in range(steps):
        # minigrad: for every layer, for every p: p.zero_grad()
        optimizer.zero_grad()

        # Operator overloading no longer has to build the graph - every op here
        # is a differentiable torch op, so autograd records it as it runs.
        predictions = model(x)

        loss = criterion(predictions, y)

        # minigrad: backward(loss), which topologically sorted the graph and
        # walked it in reverse calling each node's closure.
        loss.backward()

        # minigrad: for every p: p.data -= learning_rate * p.grad
        # Optimizer reads the .grad tensors backward() just wrote and applies
        # the update rule for us.
        optimizer.step()

        losses.append(loss.item())

    return losses


def inspect_setup(model, x, y, device):
    """Print the shapes that minigrad never had to think about."""
    with torch.no_grad():
        output = model(x)

    n_params = sum(p.numel() for p in model.parameters())

    print("Fitting y = x^2 with a 1 -> 8 -> 8 -> 1 MLP, in PyTorch")
    print(f"torch {torch.__version__}")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print()
    print(f"  {'x'.ljust(14)} {str(tuple(x.shape)).ljust(10)} {x.dtype}")
    print(f"  {'y (targets)'.ljust(14)} {str(tuple(y.shape)).ljust(10)} {y.dtype}")
    print(f"  {'predictions'.ljust(14)} {str(tuple(output.shape)).ljust(10)} {output.dtype}")
    print()
    print(f"  trainable parameters: {n_params}")
    for name, param in model.named_parameters():
        print(f"    {name:<16} {str(tuple(param.shape)):<10} requires_grad={param.requires_grad}")


def inspect_gradients(model, x, y, learning_rate=LEARNING_RATE):
    """Show forward -> loss -> backward -> .grad, stopping before optimizer.step().

    This is the part minigrad had to be explicit about at every layer: the
    gradients live in p.grad only after backward() has run, and optimizer.step()
    consumes them. Zeroing them is the job of optimizer.zero_grad(), not of
    backward().
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()

    print(f"after loss.backward(), loss = {loss.item():.6g}")
    print(f"  {'parameter'.ljust(16)} {'shape'.ljust(10)} {'grad mean':>12} {'grad absmax':>13}")

    for name, param in model.named_parameters():
        print(
            f"  {name:<16} {str(tuple(param.shape)):<10}"
            f" {param.grad.mean().item():>12.6g} {param.grad.abs().max().item():>13.6g}"
        )

    # The step is deliberately not taken here, so the .grad values above are
    # exactly what backward() produced.
    return loss


def show_predictions(model, x, y, title):
    print()
    print(title)
    print(f"  {'x':>6}  {'target':>7}  {'predicted':>9}  {'error':>7}")

    with torch.no_grad():
        preds = model(x)

    for x_i, target, pred in zip(x.flatten().tolist(), y.flatten().tolist(), preds.flatten().tolist()):
        print(f"  {x_i:6.2f}  {target:7.4f}  {pred:9.4f}  {pred - target:7.4f}")


def show_progress(losses, learning_rate):
    print(f"training for {len(losses)} steps at learning rate {learning_rate}")

    logged = range(0, len(losses), LOG_EVERY)
    for step in logged:
        print(f"  step {step:4d}  loss {losses[step]:.6g}")
    if len(losses) - 1 not in logged:
        print(f"  step {len(losses) - 1:4d}  loss {losses[-1]:.6g}")


def main():
    device = get_device()

    x, y = make_dataset(device)
    model = build_model(device)

    inspect_setup(model, x, y, device)

    losses = train(model, x, y, LEARNING_RATE, STEPS)
    show_progress(losses, LEARNING_RATE)

    show_predictions(model, x, y, "after training:")

    print()
    print("gradient inspection (the same pipeline, stopping before optimizer.step()):")
    inspect_gradients(model, x, y)

    with torch.no_grad():
        mean_error = (model(x) - y).abs().mean().item()

    print()
    print(f"learning rate {LEARNING_RATE}: {losses[0]:.6g} -> {losses[-1]:.6g}")
    print(f"mean absolute prediction error: {mean_error:.6g}")


if __name__ == "__main__":
    main()