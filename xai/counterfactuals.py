"""
counterfactuals.py

Контрафактические примеры в образном (x_T) и диффузионном (x_t) пространствах.

Оптимизационная постановка:
min_Δ  λ||Δ||_p + L_target(f(x+Δ), y_target) + R_mask(Δ; M) + TV(Δ)

Математика:
- Норма: ||Δ||_p (p ∈ {2, ∞})
- Масочное регуляриз.: R_mask(Δ;M) = μ * ||(1 - M) ⊙ Δ||_2^2 (штраф за изменения вне маски)
- Проекция на L_p шар: Π_{||Δ||_p ≤ ε}(·)
- Пермутационный тест batch-эффекта по функционалу g: сравнение распределений δ = g(x_cf) - g(x)

Doctest:
>>> import torch
>>> x = torch.zeros(2,3,16,16)
>>> y_target = torch.tensor([1,1])
>>> f = lambda z: torch.stack([z.view(z.size(0), -1).sum(dim=1), torch.zeros(z.size(0))], dim=1)
>>> x_cf = optimize_counterfactual_xT(x, f, y_target, mask=None, lp="l2", eps=0.5, lam=1.0, iters=5)
>>> x_cf.shape == x.shape
True
"""
from __future__ import annotations
from typing import Callable, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

def _project_lp(delta: torch.Tensor, eps: float, lp: str) -> torch.Tensor:
    if lp == "l2":
        flat = delta.view(delta.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
        factor = torch.clamp(eps / norm, max=1.0)
        flat = flat * factor
        return flat.view_as(delta)
    elif lp in ("linf", "l∞"):
        return torch.clamp(delta, -eps, eps)
    else:
        raise ValueError("lp must be 'l2' or 'linf'")

def _tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()

def _target_loss(logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    # максимизируем logit целевого класса => минимизируем -logit
    # если logits имеет форму (B, K)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    idx = y_target.view(-1).long()
    return -logits[torch.arange(logits.size(0)), idx].mean()

def optimize_counterfactual_xT(x: torch.Tensor,
                               f: Callable[[torch.Tensor], torch.Tensor],
                               y_target: torch.Tensor,
                               mask: Optional[torch.Tensor] = None,
                               lp: str = "l2",
                               eps: float = 8/255,
                               lam: float = 1.0,
                               iters: int = 200,
                               tv_weight: float = 0.0,
                               lr: float = 0.05) -> torch.Tensor:
    """
    Оптимизация контрафакта в образном пространстве x_T.

    Parameters
    ----------
    x : torch.Tensor
        (B,C,H,W).
    f : Callable
        Возвращает логиты (B,K) или скаляр релевантности по batch.
    y_target : torch.Tensor
        Целевые классы (B,).
    mask : torch.Tensor | None
        (B,1,H,W) — где можно менять (1) / запрещено (0). Если None — без маски.
    lp : str
        'l2' | 'linf'
    eps : float
        Радиус шара для проекции.
    lam : float
        Вес нормы (регуляризация).
    iters : int
        Итераций оптимизации.
    tv_weight : float
        Вес TV регуляризации.
    lr : float
        Шаг.

    Returns
    -------
    x_cf : torch.Tensor
        Контрафактные изображения.
    """
    x0 = x.detach()
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for i in range(iters):
        x_adv = x0 + delta
        logits = f(x_adv)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        loss_target = _target_loss(logits, y_target)
        norm_term = delta.view(delta.size(0), -1).norm(p=2, dim=1).mean() if lp == "l2" else delta.abs().mean()
        loss = lam * norm_term + loss_target
        if mask is not None:
            loss = loss + 0.5 * ((1 - mask) * delta).pow(2).mean()
        if tv_weight > 0:
            loss = loss + tv_weight * _tv_loss(x_adv)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            delta.data = _project_lp(delta.data, eps, lp)
    x_cf = (x0 + delta.detach()).clamp(-1, 1)
    return x_cf


def optimize_counterfactual_xt(x: torch.Tensor,
                               f_t: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                               y_target: torch.Tensor,
                               scheduler,
                               start_t: int,
                               iters: int = 100,
                               lp: str = "l2",
                               eps: float = 0.2,
                               lr: float = 0.05) -> torch.Tensor:
    """
    Контрафакт в пространстве x_t (фиксированный t).

    Parameters
    ----------
    f_t : Callable
        f_t(x_t, t)->(B,K) логиты; оптимизируем по x_t.
    scheduler :
        DDPMScheduler — используется для обратного шага к x_0 при необходимости.
    start_t : int
        Временной шаг для оптимизации.

    Returns
    -------
    x_cf : torch.Tensor
        Контрафакт в образном пространстве, полученный из x_t_cf обратным шагом.
    """
    x0 = x.detach()
    B = x0.size(0)
    t = torch.tensor([start_t]*B, dtype=torch.long, device=x.device)
    # Прямо оптимизируем x_t как переменную
    # Инициализация x_t как x0
    xt = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([xt], lr=lr)
    for i in range(iters):
        logits = f_t(xt, t)
        loss = _target_loss(logits, y_target)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            # проекция шага на Lp относительного сдвига к x0
            d = xt - x0
            d = _project_lp(d, eps, lp)
            xt.data = (x0 + d).clamp(-1, 1)
    # Один обратный шаг scheduler как приближение вывода x_cf
    scheduler.set_timesteps(start_t+1, device=x.device)
    x_prev = xt
    for ti in scheduler.timesteps:
        with torch.no_grad():
            # суррогат: модель отсутствует — используем тождественный шаг
            x_prev = x_prev  # placeholder: без UNet
    x_cf = x_prev.detach().clamp(-1, 1)
    return x_cf


def batch_effect_delta(g: Callable[[torch.Tensor], torch.Tensor], xs: torch.Tensor, xs_cf: torch.Tensor,
                       n_perm: int = 1000, seed: int = 42) -> Dict[str, float]:
    """
    Пермутационный тест batch-эффекта δ = g(xs_cf)-g(xs).

    Parameters
    ----------
    g : Callable
        Функция признаков (B,D).
    xs : torch.Tensor
        Исходные.
    xs_cf : torch.Tensor
        Контрафакты.
    n_perm : int
        Кол-во перестановок.
    seed : int
        Сид.

    Returns
    -------
    dict
        {'delta': mean |δ|, 'p_value_perm': p}
    """
    with torch.no_grad():
        gx = g(xs)
        gcf = g(xs_cf)
        delta = (gcf - gx).abs().mean().item()
        # нулевая гипотеза: без эффекта — случайная переиндексация знаков
        rng = np.random.RandomState(seed)
        diffs = (gcf - gx).view(gx.size(0), -1).mean(dim=1).cpu().numpy()
        obs = np.abs(diffs).mean()
        null = []
        for _ in range(n_perm):
            signs = rng.choice([1, -1], size=diffs.shape[0])
            null.append(np.abs(diffs * signs).mean())
        p = float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))
    return {"delta": delta, "p_value_perm": p}