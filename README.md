# Fashion-MNIST C++ Inference

本项目使用 PyTorch 训练 Fashion-MNIST 卷积神经网络，将权重导出为简单的自定义二进制格式，并使用纯 C++17 完成推理。C++ 程序只依赖标准库，不需要 LibTorch、OpenCV 或其他第三方运行时。

## 模型结构

```text
Input: 1 x 28 x 28
Conv2d(1, 8, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(8, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten
Linear(16 * 7 * 7, 64) -> ReLU
Linear(64, 10)
```

C++ 端实现了 Tensor、Conv2d、ReLU、MaxPool2d、Flatten 和 Linear。输入像素使用与 PyTorch 相同的归一化：`(pixel / 255.0 - 0.5) / 0.5`。

## 项目结构

```text
include/    C++ 头文件
src/        C++ 源文件
scripts/    PyTorch 训练和样本导出脚本
test/       C++ 单元测试和 Python/C++ 一致性测试
```

## Python 环境

使用 Python 自带的 `venv` 创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

训练依赖为 PyTorch 和 torchvision。训练脚本会依次尝试使用 CUDA、Apple Metal/MPS 和 CPU。

## 训练与模型导出

```bash
python scripts/train.py \
  --epochs 8 \
  --batch-size 64 \
  --lr 0.001 \
  --output model.bin \
  --checkpoint model.pth
```

`model.bin` 供 C++ 加载，`model.pth` 用于 Python/C++ 一致性测试。数据集、模型文件和 checkpoint 默认不会加入 Git。

## 构建

```bash
cmake -S . -B build
cmake --build build
```

## 导出测试图片

输入文件是包含 784 个 `0～255` 灰度整数的文本文件。可以从 Fashion-MNIST 测试集导出一张图片：

```bash
python scripts/export_sample.py --index 0 --output sample.txt
```

## C++ 推理

```bash
./build/fashion_mnist --model model.bin --input sample.txt
```

输出 logits：

```bash
./build/fashion_mnist --model model.bin --input sample.txt --logits
```

## 测试

运行全部 C++ 测试：

```bash
ctest --test-dir build --output-on-failure
```

检查 Python 脚本：

```bash
python -m py_compile scripts/*.py test/compare_inference.py
```

比较同一模型在 PyTorch 和 C++ 中的 logits：

```bash
python test/compare_inference.py \
  --checkpoint model.pth \
  --model model.bin \
  --executable build/fashion_mnist
```

一致性测试默认比较 5 张图片，要求最大绝对误差小于 `1e-4`，并检查预测类别相同。

## 模型文件格式

模型以 little-endian 保存，文件头为 `FMN1`，随后按固定顺序保存 8 个 tensor：

```text
conv1.weight
conv1.bias
conv2.weight
conv2.bias
fc1.weight
fc1.bias
fc2.weight
fc2.bias
```

每个 tensor 的布局为：

```text
uint32 ndim
uint32 shape[ndim]
float32 data[numel]
```
