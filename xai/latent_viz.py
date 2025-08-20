"""
latent_viz.py

Извлечение латентов из UNet и их проекция в 2D для визуализации и кластерных метрик.

Режимы:
- mode="bottleneck": фичи из mid_block (бутылочное горлышко), усреднение по пространству.
- mode="xt": x_t, сгенерированный по фиксированному шуму/альфе из scheduler (линейная формула).

Проекция:
- PCA -> (при необходимости) t-SNE или UMAP.
- Фиксируем сиды, масштабируем StandardScaler.

Метрики кластеризации:
- silhouette score
- Davies-Bouldin index (dbi)

Doctest:
>>> import torch
>>> from diffusers import UNet2DModel, DDPMScheduler
>>> from model_ddpm import build_unet, build_scheduler, AttentionHooks
>>> unet = build_unet(img_size=32, base_ch=32)
>>> sched = build_scheduler()
>>> x = torch.randn(4,3,32,32)
>>> feats = extract_latents(unet, x, t_list=[10], mode="bottleneck")
>>> list(feats.keys()) == [10]
True
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Опциональный импорт umap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap не установлен, UMAP проекция недоступна")

from .model_ddpm import AttentionHooks

def extract_latents(unet, x: torch.Tensor, t_list: list, mode: str = "bottleneck|xt") -> dict:
    """
    Извлекает латенты по списку t.

    Parameters
    ----------
    unet : UNet2DModel
    x : torch.Tensor
        (B,C,H,W) в [-1,1].
    t_list : list
        Таймстепы (int).
    mode : str
        "bottleneck" или "xt".

    Returns
    -------
    dict
        {t: X_t (B,D)} — плоские признаки по batch.
    """
    device = x.device
    B, C, H, W = x.shape
    mode = mode.split("|")[0] if "|" in mode else mode
    outputs = {}
    if mode == "bottleneck":
        hooks = AttentionHooks(unet)
        for t in t_list:
            tt = torch.tensor([t]*B, dtype=torch.long, device=device)
            with hooks.capture():
                _ = unet(x, tt)
            fm = hooks.get_feature_maps().get("mid_block", None)
            if fm is None:
                # fallback: используем выход UNet как фичи
                y = unet(x, tt).sample
                fm = y
            # усреднение по пространству
            feats = fm.to(x.device).float().mean(dim=(2,3))
            outputs[int(t)] = feats.detach().cpu().numpy()
    elif mode == "xt":
        # Используем x_t = α_t x + σ_t ε (псевдо, без scheduler — берём линейную смесь)
        for t in t_list:
            alpha = 1.0 - float(t) / (max(t_list) + 1e-8)
            eps = torch.zeros_like(x)
            xt = alpha * x + (1 - alpha) * eps
            outputs[int(t)] = xt.view(B, -1).detach().cpu().numpy()
    else:
        raise ValueError("mode must be 'bottleneck' or 'xt'")
    return outputs


def project_latents(X: np.ndarray, method: str = "pca|tsne|umap", pca_dim: int = 64, out_dim: int = 2, seed: int = 42):
    """
    Проецирует латенты в out_dim.

    Parameters
    ----------
    X : np.ndarray
        (N,D).
    method : str
        "pca"|"tsne"|"umap"
    pca_dim : int
        Предварительная PCA.
    out_dim : int
        Размер проекции.
    seed : int
        Сид.

    Returns
    -------
    Y : np.ndarray
        (N,out_dim)
    """
    method = method.split("|")[0] if "|" in method else method
    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    D = Xs.shape[1]
    if D > pca_dim:
        Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(Xs)
    else:
        Xp = Xs
    if method == "pca":
        Y = PCA(n_components=out_dim, random_state=seed).fit_transform(Xp)
    elif method == "tsne":
        Y = TSNE(n_components=out_dim, random_state=seed, init="pca", learning_rate="auto").fit_transform(Xp)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP недоступен. Установите: pip install umap-learn")
        reducer = umap.UMAP(n_components=out_dim, random_state=seed, metric="euclidean")
        Y = reducer.fit_transform(Xp)
    else:
        raise ValueError("Unknown method")
    return Y


def cluster_metrics(Y: np.ndarray, labels) -> dict:
    """
    Метрики кластеризации.

    Returns
    -------
    dict
        {'silhouette': float, 'dbi': float}
    """
    Y = np.asarray(Y)
    labels = np.asarray(labels)
    res = {}
    if len(np.unique(labels)) > 1 and len(Y) > len(np.unique(labels)):
        res["silhouette"] = float(silhouette_score(Y, labels))
        res["dbi"] = float(davies_bouldin_score(Y, labels))
    else:
        res["silhouette"] = float("nan")
        res["dbi"] = float("nan")
    return res