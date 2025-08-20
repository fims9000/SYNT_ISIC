"""
datasets.py

Загрузка дерматоскопического датасета или синтетических плейсхолдеров.
Ожидается структура:
root/
  train/
    MEL/*.png ...
    NV/*.png ...
    ...
  val/
    ...
  test/
    ...

Изображения: 3x128x128 (приводятся к этому размеру).
Классы: ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"].

Нормализация: в [-1,1] через mean=0.5,std=0.5 по каналам.
Аугментации отключены по умолчанию.

Если датасет недоступен:
- По умолчанию выбрасывается ошибка.
- Опционально создаются синтетические данные, если:
  1) установлена переменная окружения DERM_SYNTHETIC=1
  2) или split == 'synthetic' (специальный режим для тестов).

Doctest (synthetic):
>>> import os
>>> os.environ["DERM_SYNTHETIC"]="1"
>>> ds = load_derm_dataset("/non/existent", "train", img_size=64)
>>> from torch.utils.data import DataLoader
>>> dls = make_dataloaders("/non/existent", img_size=64, batch_size=4, num_workers=0)
>>> xb, yb = next(iter(dls["train"]))
>>> xb.shape[1:] == (3,64,64)
True
"""
from __future__ import annotations
import os
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from utils import set_seeds

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

__all__ = [
    "set_seeds",
    "load_derm_dataset",
    "make_dataloaders",
    "CLASSES",
    "CLASS_TO_IDX",
]


class DermFolder(Dataset):
    """
    Простой датасет из папок с классами.
    """

    def __init__(self, root: str, split: str, img_size: int = 128):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.samples: List[Tuple[str, int]] = []
        self._build_index()
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _build_index(self):
        split_dir = os.path.join(self.root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                # пропускаем отсутствующий класс
                continue
            files = sorted(
                glob.glob(os.path.join(cls_dir, "*.png"))
                + glob.glob(os.path.join(cls_dir, "*.jpg"))
                + glob.glob(os.path.join(cls_dir, "*.jpeg"))
            )
            for fp in files:
                self.samples.append((fp, CLASS_TO_IDX[cls]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


class SyntheticDerm(Dataset):
    """
    Синтетический датасет для проверки пайплайна.
    Генерирует простые узоры с разной текстурой/цветом для классов.

    Parameters
    ----------
    n_per_class : int
        Количество изображений на класс.
    img_size : int
        Размер (HxW).
    seed : int
        Сид генератора.
    """

    def __init__(self, n_per_class: int = 200, img_size: int = 128, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.n_per_class = n_per_class
        self.img_size = img_size
        self.n_classes = len(CLASSES)
        self.total = self.n_classes * n_per_class

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        c = idx // self.n_per_class
        rng = np.random.RandomState(idx)
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # Разные паттерны для классов
        if c == CLASS_TO_IDX["MEL"]:
            # тёмные пятна
            img += rng.uniform(0.1, 0.3)
            for _ in range(5):
                r = rng.randint(5, 15)
                cx = rng.randint(r, self.img_size - r)
                cy = rng.randint(r, self.img_size - r)
                yy, xx = np.ogrid[:self.img_size, :self.img_size]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                img[mask] = rng.uniform(0.0, 0.2)
        elif c == CLASS_TO_IDX["NV"]:
            # гладкий коричневый
            img[..., 0] = rng.uniform(0.3, 0.5)
            img[..., 1] = rng.uniform(0.2, 0.4)
            img[..., 2] = rng.uniform(0.1, 0.3)
        elif c == CLASS_TO_IDX["BCC"]:
            # полосы
            for i in range(self.img_size):
                img[i, :, :] = (i % 10) / 15.0
        elif c == CLASS_TO_IDX["AKIEC"]:
            img += rng.uniform(0.4, 0.6)
        elif c == CLASS_TO_IDX["BKL"]:
            img += rng.uniform(0.2, 0.4)
            img[:, ::2, :] += 0.1
        elif c == CLASS_TO_IDX["DF"]:
            img += rng.uniform(0.1, 0.2)
            img[::4, ::4, :] += 0.5
        elif c == CLASS_TO_IDX["VASC"]:
            img[..., 0] += 0.6
            img[..., 1] += 0.1

        img = np.clip(img, 0.0, 1.0)
        # в [-1,1]
        img = (img - 0.5) / 0.5
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        y = c
        return x, y


def _maybe_synthetic(split: str) -> bool:
    if split.lower() == "synthetic":
        return True
    return os.environ.get("DERM_SYNTHETIC", "0") == "1"


def load_derm_dataset(root: str, split: str, img_size: int = 128) -> torch.utils.data.Dataset:
    """
    Загружает датасет дерматоскопии.

    Parameters
    ----------
    root : str
        Корневая директория.
    split : str
        'train'|'val'|'test'| 'synthetic' (спец. режим).
    img_size : int
        Размер изображений.

    Returns
    -------
    torch.utils.data.Dataset

    Notes
    -----
    По умолчанию, при отсутствии данных — ошибка.
    Если DERM_SYNTHETIC=1 или split='synthetic' — отдаёт SyntheticDerm.
    """
    if _maybe_synthetic(split) or not os.path.isdir(os.path.join(root, split)):
        if os.environ.get("DERM_SYNTHETIC", "0") == "1" or split.lower() == "synthetic":
            return SyntheticDerm(n_per_class=200, img_size=img_size, seed=42)
        raise FileNotFoundError(f"Dataset split not found and synthetic disabled: {root}/{split}")
    return DermFolder(root, split, img_size=img_size)


def make_dataloaders(root: str, img_size: int = 128, batch_size: int = 32, num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Создаёт DataLoader'ы для train/val/test.

    Parameters
    ----------
    root : str
        Путь к датасету.
    img_size : int
        Размер изображений.
    batch_size : int
        Размер батча.
    num_workers : int
        Воркеры.

    Returns
    -------
    dict
        {'train': DL, 'val': DL, 'test': DL}
    """
    splits = ["train", "val", "test"]
    dsets = {s: load_derm_dataset(root, s, img_size=img_size) for s in splits}
    dls = {
        s: DataLoader(
            dsets[s],
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for s in splits
    }
    return dls