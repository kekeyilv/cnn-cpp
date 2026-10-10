import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from network import FashionCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and C++ inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--model", type=Path, default=Path("model.bin"))
    parser.add_argument(
        "--executable", type=Path, default=Path("build/fashion_mnist")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def parse_cpp_output(output: str) -> tuple[list[float], int]:
    logits = None
    prediction = None

    for line in output.splitlines():
        if line.startswith("Logits:"):
            logits = [float(value) for value in line.removeprefix("Logits:").split()]
        elif line.startswith("Prediction:"):
            prediction = int(line.split()[1])

    if logits is None or len(logits) != 10 or prediction is None:
        raise RuntimeError(f"unexpected C++ output:\n{output}")
    return logits, prediction


def export_image(image: object, path: Path) -> None:
    pixels = list(image.tobytes())
    path.write_text(" ".join(str(pixel) for pixel in pixels) + "\n")


def main() -> None:
    args = parse_args()
    model = FashionCNN()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    largest_error = 0.0
    with tempfile.TemporaryDirectory() as temporary_dir:
        sample_path = Path(temporary_dir) / "sample.txt"

        for index in range(args.samples):
            image, _ = dataset[index]
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                pytorch_logits = model(input_tensor).squeeze(0)

            export_image(image, sample_path)
            result = subprocess.run(
                [
                    str(args.executable),
                    "--model",
                    str(args.model),
                    "--input",
                    str(sample_path),
                    "--logits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            cpp_logits, cpp_prediction = parse_cpp_output(result.stdout)
            errors = [
                abs(pytorch - cpp)
                for pytorch, cpp in zip(pytorch_logits.tolist(), cpp_logits)
            ]
            max_error = max(errors)
            largest_error = max(largest_error, max_error)
            pytorch_prediction = int(pytorch_logits.argmax().item())

            if max_error >= args.tolerance:
                raise RuntimeError(
                    f"sample {index}: max absolute error {max_error:.8g} "
                    f"exceeds {args.tolerance:.8g}"
                )
            if pytorch_prediction != cpp_prediction:
                raise RuntimeError(
                    f"sample {index}: prediction mismatch "
                    f"({pytorch_prediction} != {cpp_prediction})"
                )
            print(
                f"Sample {index}: prediction={cpp_prediction}, "
                f"max_error={max_error:.8g}"
            )

    print(
        f"Compared {args.samples} samples successfully; "
        f"largest_error={largest_error:.8g}"
    )


if __name__ == "__main__":
    main()
