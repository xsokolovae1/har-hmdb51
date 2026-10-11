from pathlib import Path
import cv2
import numpy as np
import torch

N_FRAMES = 16
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_frames(path, n=N_FRAMES, size=SIZE):
    files = sorted(Path(path).glob("*.jpg"), key=lambda f: int(f.stem))
    if not files:
        raise RuntimeError(f"no frames in {path}")
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    frames = [cv2.imread(str(files[i])) for i in idx]
    return np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size)) for f in frames])


def preprocess(frames):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD