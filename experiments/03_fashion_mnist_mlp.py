"""
Experiment 03 - classifying Fashion-MNIST with an MLP, in PyTorch.

What we are learning
    The previous experiments fit a one-dimensional curve with a tiny network.
    This one stays small - three linear layers, no convolution - but does a real
    image task: sorting 28x28 grayscale photos of clothing into 10 categories
    (t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag,
    ankle boot). It is the smallest setup where the whole supervised pipeline
    shows up: a train/validation/test split, mini-batches, shuffled data,
    cross-entropy loss, and checkpoint selection.

Why an MLP and not a CNN
    An MLP sees each pixel as an independent number and has no notion of a
    pixel's neighbours, so it is a deliberately weak image model. That is the
    point here: the architecture is plain, so the experiment is about the
    training machinery, not about squeezing out accuracy. A convolutional
    network would be the natural next step.

Run with:
    .venv/bin/python experiments/03_fashion_mnist_mlp.py --epochs 5

Fashion-MNIST is downloaded on first use into the project-local data/ folder.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 0
ARCHITECTURE = [784, 256, 128, 10]
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 5
VALIDATION_FRACTION = 0.1

# Pixel statistics of the Fashion-MNIST training set, in [0, 1]. Normalizing
# with them keeps the inputs roughly zero-mean and unit-variance, which makes
# the first layer's gradients behave. Data is downloaded into a project-local
# folder (kept out of git); it is not committed.
MEAN = (0.2860,)
STD = (0.3530,)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def get_device():
    """Prefer the working GPU, otherwise the CPU.

    On AMD, PyTorch reuses CUDA's API names for ROCm, so torch.cuda is the
    right module for either vendor. We pin the device to index 0 on purpose:
    this machine exposes a second "CUDA" device that is not the GPU we want, so
    selecting cuda:1 would be a bug.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_fashion_mnist(data_dir=DATA_DIR):
    """Return the official training and test datasets, downloading if needed.

    ToTensor() converts a PIL image to a [1, 28, 28] float tensor scaled to
    [0, 1]; Normalize() then standardizes it. The test set is loaded here but
    only used once, at the very end - it must stay untouched during training
    and model selection.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    data_dir = Path(data_dir)
    try:
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    except Exception as error:  # noqa: BLE001 - surface the real cause, clearly
        raise RuntimeError(
            f"Could not download or load Fashion-MNIST under {data_dir}. "
            f"Check the network connection and write permissions. "
            f"Original error: {error!r}"
        ) from error

    return train_dataset, test_dataset


def split_train_validation(train_dataset, validation_fraction=VALIDATION_FRACTION, seed=SEED):
    """Carve a validation set out of the official training set.

    Validation examples are held out from gradient updates; they are only used
    to pick the best model, so they act as an honest dry run for the test set.
    The split is seeded, so every run uses exactly the same examples.
    """
    n_total = len(train_dataset)
    n_val = int(round(n_total * validation_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [n_train, n_val], generator=generator
    )

    return train_subset, val_subset


def make_loaders(train_subset, val_subset, test_dataset, seed=SEED):
    """Wrap the three splits in mini-batch loaders.

    Training data is shuffled each epoch so the batches differ and the model
    does not learn the order of the file. Validation and test data are not
    shuffled, because their order is irrelevant to a plain accuracy/loss
    average. The generator makes the shuffling reproducible.
    """
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def flatten_images(images):
    """Reshape [batch, 1, 28, 28] images into [batch, 784] rows.

    An MLP treats each example as one flat vector of independent input features.
    It has no idea that the 784 numbers are arranged as a 2D grid, so the grid
    structure is simply thrown away by flattening before the first Linear layer.
    (A CNN, by contrast, keeps the 2D layout.)
    """
    return images.view(images.size(0), -1)


class MLP(nn.Module):
    """784 -> 256 -> 128 -> 10 with ReLU between the hidden layers.

    The final layer returns 10 raw logits, one per class. Logits are unbounded
    real numbers: the larger the logit for a class, the more the model favours
    it. They are deliberately *not* probabilities, and no softmax is applied
    here, because CrossEntropyLoss expects raw logits (see the training loop).
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ARCHITECTURE[0], ARCHITECTURE[1]),
            nn.ReLU(),
            nn.Linear(ARCHITECTURE[1], ARCHITECTURE[2]),
            nn.ReLU(),
            nn.Linear(ARCHITECTURE[2], ARCHITECTURE[3]),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(device, seed=SEED):
    """Seed before constructing so the initial weights are reproducible."""
    torch.manual_seed(seed)
    model = MLP().to(device)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one pass over the training data and return the mean loss.

    The four-step pattern is the same one the earlier experiments wrote by hand:
        optimizer.zero_grad()  clears the gradients left over from the last step
                               (PyTorch accumulates into .grad by default);
        loss.backward()        builds the autograd graph backwards and fills
                               every parameter's .grad;
        optimizer.step()       applies the update rule (here Adam) using .grad.

    Note criterion receives the raw logits, not softmax probabilities. The loss
    applies log_softmax internally, so feeding it probabilities would take the
    log of a softmax a second time and give the wrong number and wrong
    gradients.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = flatten_images(images).to(device)
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
    """Return (mean loss, accuracy) over a loader. Used for validation and test.

    model.eval() turns off training-only behaviour (here it matters once we
    later add dropout/batchnorm) and no_grad() skips building the autograd
    graph, so evaluation is fast and cannot leak gradients. Validation and test
    differ only in intent: validation guides model selection, test is the final
    unbiased number that is reported once.
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = flatten_images(images).to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)

    return total_loss / n, correct / n


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train an MLP on Fashion-MNIST.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"number of training epochs (default {EPOCHS})",
    )
    args = parser.parse_args(argv)

    # Seed the global RNGs so weight init and any other randomness repeat.
    torch.manual_seed(SEED)

    device = get_device()
    print(f"torch {torch.__version__}")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    train_dataset, test_dataset = load_fashion_mnist()
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
    print(f"model: {' -> '.join(str(w) for w in ARCHITECTURE)}  ({n_params} parameters)")

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
            # Keep a CPU copy of the best weights; only these are evaluated on
            # the test set once training is over.
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
