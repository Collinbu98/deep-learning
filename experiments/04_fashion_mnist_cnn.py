"""
Experiment 04 - classifying Fashion-MNIST with a small CNN, in PyTorch.

What we are learning
    Experiment 03 flattened every 28x28 image into 784 independent numbers and
    fed it to an MLP, reaching 87.86% test accuracy in five epochs. Flattening
    throws away the fact that pixels are arranged in a grid, so the MLP cannot
    easily tell that two neighbouring pixels belong together, and it needs a
    separate weight for every pixel of every feature.

    A convolutional neural network keeps the 2D layout. A conv layer slides a
    small learnable filter across the image and reuses the same weights at every
    position, so it can detect a local pattern (an edge, a seam, a sole) no
    matter where it appears. This experiment swaps in a compact CNN and trains
    it with everything else held fixed - same data, split, loss, optimizer and
    epoch budget - to see what convolution buys us.

What a CNN layer actually does
    - A *filter* (kernel) is a small grid of weights, e.g. 3x3. It is convolved
      over the input: at each position it multiplies its weights by the local
      patch and sums, producing one number. The same filter is reused at every
      position, which is what makes a CNN far more parameter-efficient than an
      MLP on images.
    - *Channels* are parallel feature maps. The first layer has 1 input channel
      (grayscale) and learns 16 filters, so it outputs 16 *feature maps*, each
      one a response map showing where that filter fired.
    - *Padding* of 1 pixel around a 3x3 filter keeps the spatial size unchanged,
      so a conv layer maps 28x28 to 28x28.
    - *Pooling* shrinks each feature map by keeping only the strongest response
      in each 2x2 block. This halves the resolution, cutting computation and
      giving some tolerance to small shifts.
    - After two conv+pool blocks the 28x28 image has become 32 feature maps of
      7x7. The *classifier* flattens that into a 1568-vector and uses ordinary
      Linear layers to turn it into 10 class logits.

Run with:
    .venv/bin/python experiments/04_fashion_mnist_cnn.py --epochs 5

Fashion-MNIST is downloaded on first use into the project-local data/ folder.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch import nn

SEED = 0
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 5

# The conv stack maps a [1, 28, 28] image to 32 feature maps of 7x7:
#   28x28 --conv--> 28x28 --pool--> 14x14 --conv--> 14x14 --pool--> 7x7
FEATURE_MAPS = 32
FEATURE_SIZE = 7


def _load_experiment(name, filename):
    """Import a sibling experiment file whose name is not a valid module.

    Experiment 03 already solved data handling: the exact normalization, the
    seeded train/validation split and the mini-batch loaders. Reusing them here
    guarantees the CNN and the MLP are compared on byte-for-byte identical
    splits instead of two lookalike pipelines that could quietly diverge.
    """
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_mlp_experiment = _load_experiment("fashion_mnist_mlp", "03_fashion_mnist_mlp.py")

# Shared data pipeline (device choice, dataset download, split, loaders).
get_device = _mlp_experiment.get_device
load_fashion_mnist = _mlp_experiment.load_fashion_mnist
split_train_validation = _mlp_experiment.split_train_validation
make_loaders = _mlp_experiment.make_loaders
DATA_DIR = _mlp_experiment.DATA_DIR


class CNN(nn.Module):
    """Two conv+pool blocks followed by a small classifier.

    Shapes, batch size B:
        input            [B,  1, 28, 28]
        conv1 + relu     [B, 16, 28, 28]   (padding 1 preserves size)
        maxpool 2x2      [B, 16, 14, 14]
        conv2 + relu     [B, 32, 14, 14]
        maxpool 2x2      [B, 32,  7,  7]
        flatten          [B, 1568]
        fc1 + relu       [B, 128]
        fc2 (logits)     [B, 10]

    As in the MLP, the last layer returns raw logits (no softmax): the loss
    function expects unnormalized scores and applies log_softmax itself.
    """

    def __init__(self):
        super().__init__()
        # "features" is the convolutional stack: it turns pixels into a
        # spatial grid of learned feature maps.
        self.features = nn.Sequential(
            # 1 input channel -> 16 filters, each 3x3. Each of the 16 output
            # channels is one feature map. padding=1 keeps the 28x28 size.
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Halve the resolution: 28x28 -> 14x14.
            nn.MaxPool2d(kernel_size=2),
            # 16 feature maps -> 32 feature maps, still 3x3 filters.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 14x14 -> 7x7.
            nn.MaxPool2d(kernel_size=2),
        )
        # "classifier" reads the flattened feature maps and produces logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FEATURE_MAPS * FEATURE_SIZE * FEATURE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward_features(self, x):
        """Return the [B, 32, 7, 7] feature maps before the classifier.

        Kept as a separate method so the spatial shapes are easy to inspect and
        test without reaching into the Sequential internals.
        """
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_model(device, seed=SEED):
    """Seed before constructing so the initial weights are reproducible."""
    torch.manual_seed(seed)
    model = CNN().to(device)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data; returns the mean loss.

    Unlike the MLP loop, the images are NOT flattened: the CNN needs the
    [B, 1, 28, 28] grid. The learning loop itself is unchanged:
        optimizer.zero_grad()  clears gradients from the previous step,
        loss.backward()        fills every parameter's .grad via autograd,
        optimizer.step()       applies Adam using those gradients.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return (mean loss, accuracy) over a loader, for validation or test."""
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)

    return total_loss / n, correct / n


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a CNN on Fashion-MNIST.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"number of training epochs (default {EPOCHS})",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(SEED)

    device = get_device()
    print(f"torch {torch.__version__}")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    train_dataset, test_dataset = load_fashion_mnist(DATA_DIR)
    train_subset, val_subset = split_train_validation(train_dataset)
    train_loader, val_loader, test_loader = make_loaders(
        train_subset, val_subset, test_dataset
    )

    print(
        f"data: {len(train_subset)} train | {len(val_subset)} validation"
        f" | {len(test_dataset)} test  (batch size {BATCH_SIZE})"
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: conv(1->16,3x3) -> pool -> conv(16->32,3x3) -> pool"
        f" -> fc(1568->128) -> fc(128->10)  ({n_params} parameters)"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    print()
    print(f"training for {args.epochs} epochs at learning rate {LEARNING_RATE}")
    print(f"  {'epoch':>5}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " <- best"

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}"
            f"  {val_acc:>7.2%}{marker}"
        )

    print()
    print(f"selecting checkpoint from epoch {best_epoch} (val loss {best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test loss {test_loss:.4f}  test accuracy {test_acc:.2%}")


if __name__ == "__main__":
    main()
