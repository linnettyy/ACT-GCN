
import torch
import torchaudio
import torchvision

print("Torch version:", torch.__version__)  # 应输出 '2.1.0'
print("Torchaudio version:", torchaudio.__version__)  # 应输出 '2.1.0'
print("TorchVision version:", torchvision.__version__)  # 应输出 '0.16.0'
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())  # 如果返回 True，说明 GPU 可用