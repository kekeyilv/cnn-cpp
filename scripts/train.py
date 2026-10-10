import argparse
import struct
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import FashionCNN

TENSOR_NAMES = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
)


def make_data_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()

    return correct / len(loader.dataset)


def export_model(model: nn.Module, output_path: Path) -> None:
    state = model.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output:
        output.write(b"FMN1")
        for name in TENSOR_NAMES:
            tensor = state[name].detach().cpu().to(torch.float32).contiguous()
            output.write(struct.pack("<I", tensor.ndim))
            output.write(struct.pack(f"<{tensor.ndim}I", *tensor.shape))
            output.write(struct.pack(f"<{tensor.numel()}f", *tensor.view(-1).tolist()))

    verify_export(output_path, [tuple(state[name].shape) for name in TENSOR_NAMES])


def verify_export(path: Path, expected_shapes: list[tuple[int, ...]]) -> None:
    with path.open("rb") as model_file:
        if model_file.read(4) != b"FMN1":
            raise RuntimeError("invalid model magic")

        for expected_shape in expected_shapes:
            ndim = struct.unpack("<I", model_file.read(4))[0]
            shape = struct.unpack(f"<{ndim}I", model_file.read(4 * ndim))
            if shape != expected_shape:
                raise RuntimeError(
                    f"exported shape mismatch: {shape} != {expected_shape}"
                )

            numel = 1
            for dimension in shape:
                numel *= dimension
            if len(model_file.read(4 * numel)) != 4 * numel:
                raise RuntimeError("incomplete tensor data")

        if model_file.read(1):
            raise RuntimeError("unexpected data at end of model file")


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a Fashion-MNIST CNN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("model.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = make_data_loaders(args.data_dir, args.batch_size)
    model = FashionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"loss={loss:.4f}, test_accuracy={accuracy * 100:.2f}%"
        )

    final_accuracy = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {final_accuracy * 100:.2f}%")

    export_model(model, args.output)
    print(f"Exported 8 tensors to {args.output} ({args.output.stat().st_size} bytes)")
    save_checkpoint(model, args.checkpoint)
    print(f"Saved PyTorch checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
