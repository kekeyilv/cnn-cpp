import argparse
from pathlib import Path

from torchvision import datasets


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Fashion-MNIST test image")
    parser.add_argument("--output", type=Path, default=Path("sample.txt"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True)
    image, label = dataset[args.index]
    pixels = list(image.tobytes())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(" ".join(str(pixel) for pixel in pixels) + "\n")
    print(f"Exported sample {args.index}: {label} ({CLASS_NAMES[label]}) to {args.output}")


if __name__ == "__main__":
    main()
