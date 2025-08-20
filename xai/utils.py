"""
utils.py

Вспомогательные утилиты: фиксация сидов, перенос тензоров на устройство,
сохранение карт важности и оверлеев, JSON-логи, таймер.

Doctest:
>>> import torch, numpy as np
>>> from PIL import Image
>>> set_seeds(123)
>>> x = torch.rand(1,3,8,8)
>>> dev = torch.device('cpu')
>>> x2 = to_device(x, dev)
>>> assert x2.device.type == 'cpu'
>>> import tempfile, os, numpy as np
>>> d = tempfile.mkdtemp()
>>> img = (np.random.rand(128,128,3)*255).astype('uint8')
>>> hm = (np.random.rand(128,128)).astype('float32')
>>> save_heatmap(img, hm, os.path.join(d, 'hm.png'), title='baseline=zeros')
>>> over = overlay(img, hm)
>>> save_json({'a':1}, os.path.join(d, 'log.json'))
"""
from __future__ import annotations
import os
import io
import json
import time
import random
import contextlib
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

__all__ = [
    "set_seeds",
    "to_device",
    "timestamp",
    "ensure_dir",
    "save_heatmap",
    "overlay",
    "save_json",
    "timer",
]

DEFAULT_CMAP = "magma"
sns.set_context("talk")
sns.set_style("whitegrid")


def set_seeds(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Фиксирует сиды для python, numpy, torch. Включает детерминизм cudnn.

    Parameters
    ----------
    seed : int
        Начальное значение генераторов.
    deterministic_cudnn : bool
        Принудительный детерминизм cudnn.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_device(x: Any, device: torch.device) -> Any:
    """
    Рекурсивно переносит тензоры на устройство.

    Parameters
    ----------
    x : Any
        Объект/структура с Tensor.
    device : torch.device
        Назначение.

    Returns
    -------
    Any
        Та же структура на устройстве.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(el, device) for el in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def timestamp() -> str:
    """
    Возвращает строку таймстемпа YYYYmmdd_HHMMSS.

    Returns
    -------
    str
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    """
    Создаёт директорию, если её нет.
    """
    os.makedirs(path, exist_ok=True)


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    if np.allclose(hm.max(), hm.min()):
        return np.zeros_like(hm)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def overlay(img: np.ndarray | Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Накладывает карту важности на изображение.

    Parameters
    ----------
    img : np.ndarray | PIL.Image
        RGB изображение HxWx3, uint8.
    heatmap : np.ndarray
        Карта HxW в [0,1] (нормализуется внутри).
    alpha : float
        Прозрачность.

    Returns
    -------
    PIL.Image
        Оверлей.

    Notes
    -----
    Единый colormap: magma.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = np.asarray(img)
    hm = _normalize_heatmap(heatmap)
    cmap = plt.get_cmap(DEFAULT_CMAP)
    hm_color = (cmap(hm)[..., :3] * 255).astype(np.uint8)
    overlay_np = (alpha * hm_color + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay_np)


def save_heatmap(img: np.ndarray | Image.Image, heatmap: np.ndarray, path: str, title: str | None = None,
                 vmin: float = 0.0, vmax: float = 1.0) -> None:
    """
    Сохраняет heatmap и оверлей в один файл (2 subplot) с едиными шкалами.

    Parameters
    ----------
    img : np.ndarray | Image.Image
        RGB HxWx3, uint8.
    heatmap : np.ndarray
        HxW float.
    path : str
        Путь сохранения (png).
    title : str | None
        Заголовок (например, baseline=x0).
    vmin, vmax : float
        Единые шкалы.

    Returns
    -------
    None
    """
    ensure_dir(os.path.dirname(path))
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    else:
        img_np = np.asarray(img)
    hm = np.asarray(heatmap, dtype=np.float32)
    H, W = hm.shape
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    if title:
        fig.suptitle(title)
    im0 = axes[0].imshow(hm, cmap=DEFAULT_CMAP, vmin=vmin, vmax=vmax)
    axes[0].set_title("Heatmap")
    axes[0].axis("off")
    axes[1].imshow(overlay(img_np, hm))
    axes[1].set_title("Overlay")
    axes[1].axis("off")
    fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(obj: Dict[str, Any], path: str) -> None:
    """
    Сохраняет словарь в JSON.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@contextlib.contextmanager
def timer(name: str):
    """
    Простой таймер контекст-менеджер.

    Doctest:
    >>> with timer("work"):
    ...     x = sum(range(1000))
    """
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[TIMER] {name}: {dt:.3f}s")