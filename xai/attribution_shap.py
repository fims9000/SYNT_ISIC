"""
attribution_shap.py

Патч-ШЭП (SHAP) для изображений с выборкой коалиций патчей.

Математика:
- Значение Шэпли:
  φ_i(f,x) = ∑_S |S|!(n−|S|−1)!/n! [ f(x_{S ∪ {i}}) − f(x_S) ]
  где n — число признаков (патчей), S — коалиция без i.

- Эффективность (Efficiency): ∑_i φ_i ≈ f(x) − f(x_0).

Реализация:
- Разбиваем на непересекающиеся патчи patch_size x patch_size (n патчей).
- Сэмплирование случайных перестановок (пермутаций) патчей — unbiased оценка Шэпли.
- Маскирование: патчи вне S заменяются на baseline из x0 ("mean"|"zeros"|"noise").

Doctest:
>>> import torch
>>> x = torch.rand(2,3,32,32)
>>> f = lambda z: z.mean(dim=(1,2,3)) # скаляр
>>> phi_up, phi_p = shap_patches(x, f, patch_size=16, num_samples=64, baseline="mean")
>>> phi_up.shape == x.shape[:2] + (32,32)
True
"""
from __future__ import annotations
from typing import Callable, Tuple, List, Dict
import math
import numpy as np
import torch

def _baseline_from_mode(x: torch.Tensor, mode: str, seed: int = 0) -> torch.Tensor:
    mode = mode.lower()
    if mode == "zeros":
        return torch.zeros_like(x)
    if mode == "mean":
        m = x.mean(dim=(2,3), keepdim=True)
        return m.expand_as(x)
    if mode == "noise":
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        return torch.rand_like(x, generator=g) * 2 - 1
    return torch.zeros_like(x)

def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    B,C,H,W = x.shape
    assert H % patch == 0 and W % patch == 0, "H,W должны делиться на patch_size"
    ph, pw = patch, patch
    x = x.view(B, C, H//ph, ph, W//pw, pw)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, Nh, Nw, C, ph, pw
    return x  # не объединяем в n, чтобы проще картировать обратно

def _unpatchify(xp: torch.Tensor) -> torch.Tensor:
    # xp: B, Nh, Nw, C, ph, pw
    B, Nh, Nw, C, ph, pw = xp.shape
    x = xp.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Nh*ph, Nw*pw)
    return x

def _apply_mask(x: torch.Tensor, mask: torch.Tensor, baseline: torch.Tensor, patch: int) -> torch.Tensor:
    # mask: (B, Nh, Nw) bool => патчи что сохраняем из x, иначе baseline
    B,C,H,W = x.shape
    xp = _patchify(x, patch)
    bp = _patchify(baseline, patch)
    Nh, Nw = xp.shape[1], xp.shape[2]
    mask = mask.view(B, Nh, Nw, 1, 1, 1).to(x.dtype)
    mixed = mask * xp + (1 - mask) * bp
    return _unpatchify(mixed)

def shap_patches(x: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], patch_size: int = 16,
                 num_samples: int = 2048, baseline: str = "mean") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo Patch SHAP.

    Parameters
    ----------
    x : torch.Tensor
        (B,C,H,W).
    f : Callable
        Скалярная релевантность f(x)->(B,).
    patch_size : int
        Размер патча (H,W должны делиться на него).
    num_samples : int
        Число случайных перестановок.
    baseline : str
        "mean"|"zeros"|"noise".

    Returns
    -------
    phi_upscaled : torch.Tensor
        (B,C,H,W) с распределением φ по пикселям (равномерно внутри патча).
    phi_patches : torch.Tensor
        (B, Nh, Nw) — φ для патчей.

    Проверка эффективности:
    ∑_i φ_i ≈ f(x) − f(x_0) (контролируется на уровне патчей).
    """
    device = x.device
    B,C,H,W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    Nh, Nw = H // patch_size, W // patch_size
    n = Nh * Nw
    x0 = _baseline_from_mode(x, baseline)
    phi = torch.zeros(B, Nh, Nw, device=device)

    # Перестановки по патчам: для каждого сэмпла — своя пермутация
    # Накапливаем маргинальные вклады
    for s in range(num_samples):
        # одна пермутация для всех в batch (можно сделать индивидуально — дороже)
        order = torch.randperm(n, device=device)
        mask = torch.zeros(B, Nh, Nw, dtype=torch.bool, device=device)
        fx_prev = f(_apply_mask(x, mask, x0, patch_size))
        if fx_prev.ndim > 1:
            fx_prev = fx_prev.view(fx_prev.shape[0], -1).mean(dim=1)
        for k in range(n):
            idx = order[k].item()
            i = idx // Nw
            j = idx % Nw
            mask[:, i, j] = True
            fx_cur = f(_apply_mask(x, mask, x0, patch_size))
            if fx_cur.ndim > 1:
                fx_cur = fx_cur.view(fx_cur.shape[0], -1).mean(dim=1)
            contrib = fx_cur - fx_prev  # (B,)
            phi[:, i, j] += contrib
            fx_prev = fx_cur

    phi = phi / float(num_samples)
    # upsample: равномерно распределим вклад патча по пикселям
    phi_patch_map = phi.view(B, Nh, Nw, 1, 1).expand(B, Nh, Nw, patch_size, patch_size)
    phi_up = phi_patch_map.reshape(B, Nh*patch_size, Nw*patch_size)
    phi_up = phi_up.unsqueeze(1).expand(B, C, H, W)

    # Эффективность на уровне batch (по патчам)
    with torch.no_grad():
        fx = f(x)
        fx0 = f(x0)
        if fx.ndim > 1:
            fx = fx.view(fx.shape[0], -1).mean(dim=1)
        if fx0.ndim > 1:
            fx0 = fx0.view(fx0.shape[0], -1).mean(dim=1)
        lhs = phi.view(B, -1).sum(dim=1)
        rhs = fx - fx0
        err = (lhs - rhs).abs().mean().item()
        tol = max(0.05, 0.5 / num_samples**0.5)
        if err > tol:
            print(f"[SHAP] Efficiency deviation mean |sum(phi)-(f(x)-f(x0))|={err:.4f} > tol={tol:.4f}")

    return phi_up, phi


def time_shap(x: torch.Tensor, f_t, t_list: list, patch_size: int = 16,
              num_samples: int = 1024, baseline: str = "mean"):
    """
    Временная SHAP-агрегация по t.

    Returns
    -------
    dict
        {t: (phi_up, phi_patches), "agg": aggregated_up}
    """
    results = {}
    ups = []
    for t in t_list:
        t_t = torch.tensor([t]*x.shape[0], device=x.device, dtype=torch.long)
        def f_wrap(xi): return f_t(xi, t_t)
        phi_up, phi_p = shap_patches(x, f_wrap, patch_size=patch_size, num_samples=num_samples, baseline=baseline)
        results[int(t)] = (phi_up, phi_p)
        ups.append(phi_up)
    results["agg"] = torch.stack(ups, dim=0).mean(dim=0)
    return results