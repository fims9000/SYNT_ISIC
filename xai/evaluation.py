"""
evaluation.py

Проверки XAI:
- sanity_checks: деградация карт при рандомизации слоев
- check_ig_completeness: IG полнота
- check_shap_efficiency: SHAP эффективность
- methods_consistency: согласованность методов (Спирмен)
- counterfactual_stability: устойчивость δ

Математика:
- Completeness IG: sum_i IG_i ≈ f(x)-f(x0)
- Efficiency SHAP: sum_i φ_i ≈ f(x)-f(x0)
- Spearman ρ: корреляция рангов между методами

Doctest:
>>> import torch
>>> import numpy as np
>>> from scipy.stats import rankdata
>>> ranks_list = [np.array([1,2,3]), np.array([1,3,2])]
>>> r = methods_consistency(ranks_list)
>>> isinstance(r, float)
True
"""
from __future__ import annotations
from typing import Callable, List, Dict, Any
import copy
import warnings
import numpy as np
import torch
from scipy.stats import spearmanr

def sanity_checks(model: torch.nn.Module, attribution_fn_list: List[Callable], x: torch.Tensor, f: Callable) -> Dict[str, Any]:
    """
    Санити-чек: рандомизация слоёв должна снижать качество карт.

    Procedure
    ---------
    1) Базовые карты: для текущей модели.
    2) Рандомизируем веса верхних слоёв (поверхностно) и снимаем карты.
    3) Считаем корреляции с базовыми — должны падать.

    Returns
    -------
    dict
        { 'corr_drop_mean': float, 'per_method': {i: corr} }
    """
    model_eval = model.eval()
    base_maps = []
    with torch.no_grad():
        for attr in attribution_fn_list:
            base_maps.append(attr(x))
    # рандомизация поверхностных слоёв (последний блок)
    model_rand = copy.deepcopy(model_eval)
    for name, p in model_rand.named_parameters():
        if any(k in name for k in ["up_blocks.0", "conv_out", "proj_out"]):
            torch.nn.init.normal_(p, std=0.1)
    rand_maps = []
    with torch.no_grad():
        for attr in attribution_fn_list:
            rand_maps.append(attr(x))
    # усредненная корреляция Спирмена по картам
    per_method = {}
    drops = []
    for i, (A, B) in enumerate(zip(base_maps, rand_maps)):
        a = A.view(A.size(0), -1).mean(dim=0).cpu().numpy()
        b = B.view(B.size(0), -1).mean(dim=0).cpu().numpy()
        rho = spearmanr(a, b).correlation
        per_method[i] = float(rho)
        drops.append(1.0 - float(max(rho, 0.0)))
    return {"corr_drop_mean": float(np.mean(drops)), "per_method": per_method}


def check_ig_completeness(IG: torch.Tensor, f_x: torch.Tensor, f_x0: torch.Tensor, tol: float = 0.05) -> float:
    """
    Проверка полноты IG: |sum(IG) - (f(x)-f(x0))| <= tol.

    Returns
    -------
    float
        Средняя абсолютная невязка по batch.
    """
    if f_x.ndim > 1:
        f_x = f_x.view(f_x.shape[0], -1).mean(dim=1)
    if f_x0.ndim > 1:
        f_x0 = f_x0.view(f_x0.shape[0], -1).mean(dim=1)
    lhs = IG.view(IG.shape[0], -1).sum(dim=1)
    rhs = f_x - f_x0
    err = (lhs - rhs).abs().mean().item()
    if err > tol:
        warnings.warn(f"IG completeness error {err:.4f} > tol={tol}")
    return err


def check_shap_efficiency(phi: torch.Tensor, f_x: torch.Tensor, f_x0: torch.Tensor, tol: float = 0.05) -> float:
    """
    Проверка эффективности SHAP: |sum(phi) - (f(x)-f(x0))| <= tol.

    Returns
    -------
    float
        Средняя абсолютная невязка по batch.
    """
    if f_x.ndim > 1:
        f_x = f_x.view(f_x.shape[0], -1).mean(dim=1)
    if f_x0.ndim > 1:
        f_x0 = f_x0.view(f_x0.shape[0], -1).mean(dim=1)
    lhs = phi.view(phi.shape[0], -1).sum(dim=1)
    rhs = f_x - f_x0
    err = (lhs - rhs).abs().mean().item()
    if err > tol:
        warnings.warn(f"SHAP efficiency error {err:.4f} > tol={tol}")
    return err


def methods_consistency(ranks_list: List[np.ndarray]) -> float:
    """
    Spearman корреляция между средними рангами методов.

    Parameters
    ----------
    ranks_list : list of np.ndarray
        Списки рангов (чем меньше, тем важнее).

    Returns
    -------
    float
        Средний Спирмен между всеми парами.
    """
    n = len(ranks_list)
    if n < 2:
        return float("nan")
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            r = spearmanr(ranks_list[i], ranks_list[j]).correlation
            vals.append(r)
    return float(np.nanmean(vals))


def counterfactual_stability(g: Callable[[torch.Tensor], torch.Tensor],
                             xs: torch.Tensor,
                             xs_cf: torch.Tensor,
                             xs_ctrl: torch.Tensor) -> Dict[str, float]:
    """
    Устойчивость δ: сравниваем |g(xs_cf)-g(xs)| и |g(xs_ctrl)-g(xs)|.

    Returns
    -------
    dict
        {'delta_cf': float, 'delta_ctrl': float, 'ratio': float}
    """
    with torch.no_grad():
        gx = g(xs)
        gcf = g(xs_cf)
        gctrl = g(xs_ctrl)
        d_cf = (gcf - gx).abs().mean().item()
        d_ctrl = (gctrl - gx).abs().mean().item()
        ratio = d_cf / (d_ctrl + 1e-8)
    return {"delta_cf": d_cf, "delta_ctrl": d_ctrl, "ratio": ratio}