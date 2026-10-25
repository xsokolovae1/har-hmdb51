"""Spoločné funkcie pre všetky notebooky projektu.

Načítanie snímok, predspracovanie pre siete a jednotný protokol hodnotenia.
Rovnaké funkcie používa aj skript pre kameru, takže predspracovanie
pri trénovaní a pri živom spustení je vždy rovnaké.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["eat", "drink"]  # index triedy = label (0 = eat, 1 = drink)
N_FRAMES = 16               # počet snímok na klip
SIZE = 224                  # veľkosť snímky (vstup ResNet18)

# normalizácia, s ktorou bol ResNet18 trénovaný na ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- snímky

def sample_frames(frames, n=N_FRAMES, size=SIZE):
    """Z postupnosti RGB snímok vyberie n rovnomerne rozložených a zmení ich veľkosť.

    Používa sa pri datasete aj pri kamere -> (n, size, size, 3) uint8.
    """
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([cv2.resize(frames[i], (size, size)) for i in idx])


def read_frames(path, n=N_FRAMES, size=SIZE):
    """Načíta klip z priečinka so snímkami (*.jpg) -> (n, size, size, 3) uint8 RGB."""
    files = sorted(Path(path).glob("*.jpg"), key=lambda f: int(f.stem))
    if not files:
        raise RuntimeError(f"V priečinku {path} nie sú žiadne snímky")
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    frames = [cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB) for i in idx]
    return np.stack([cv2.resize(f, (size, size)) for f in frames])


def to_imagenet_tensor(frames):
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float, normalizované ako ImageNet."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def to_video_tensor(frames, video_transform):
    """(T, H, W, 3) uint8 -> (1, 3, T, h, w) pre video sieť (napr. r2plus1d).

    video_transform = weights.transforms() danej siete (resize, crop, normalizácia).
    """
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)   # (T, 3, H, W)
    return video_transform(x).unsqueeze(0)              # (1, 3, T, h, w)


# ---------------------------------------------------------------- hodnotenie

def make_splits(df, n_splits=10, test_size=0.3):
    """Opakované delenie na train/test (70/30) podľa zdrojového videa.

    Klipy z jedného filmu sú vždy celé buď v train, alebo v test,
    inak by model bol testovaný na takmer rovnakých záberoch, aké videl pri tréningu.
    Vracia (seed, train_idx, test_idx).
    """
    for seed in range(n_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=df.source))
        assert not set(df.source.iloc[tr]) & set(df.source.iloc[te])  # žiadny únik
        yield seed, tr, te


def metrics(y_true, y_pred):
    """Metriky, ktoré reportujeme pre každý model."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),  # priemer recall cez triedy
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def svm_search(X, y, groups):
    """SVM s výberom kernelu a hyperparametrov krížovou validáciou (len na trénovacích dátach)."""
    grid = [
        {"clf__kernel": ["linear"], "clf__C": [0.001, 0.01, 0.1, 1]},
        {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 1e-3, 1e-4]},
    ]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(class_weight="balanced"))])  # váhy tried kvôli nevyváženosti
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy")
    gs.fit(X, y, groups=groups)
    return gs


def evaluate_svm(X, df, n_splits=10):
    """Vyhodnotí SVM na opakovaných deleniach.

    Vracia tabuľku metrík pre každé delenie a súčet predikcií (pre confusion matrix).
    """
    y, groups = df.label.values, df.source.values
    rows, y_all, p_all = [], [], []
    for seed, tr, te in make_splits(df, n_splits):
        gs = svm_search(X[tr], y[tr], groups[tr])
        pred = gs.predict(X[te])
        rows.append({"split": seed, **metrics(y[te], pred), "params": gs.best_params_})
        y_all.append(y[te])
        p_all.append(pred)
    return pd.DataFrame(rows), np.concatenate(y_all), np.concatenate(p_all)


def summarize(results, name):
    """Priemer ± smerodajná odchýlka cez delenia -> jeden riadok tabuľky."""
    m = results[["acc", "bal_acc", "f1"]]
    return pd.Series({c: f"{m[c].mean():.3f} ± {m[c].std():.3f}" for c in m}, name=name)
