"""
attribution_ig.py

Интегрированные градиенты (IG) для изображения.

Математика:
- IG определяются как
  IG_i(x) = (x_i - x_{0,i}) * ∫_0^1 ∂f(x_0 + α(x - x_0)) / ∂x_i dα
  Аппроксимация трапецеидальная/средняя по m точкам на траектории.

- Полнота (Completeness): sum_i IG_i ≈ f(x) - f(x_0).

Doctest:
>>> import torch
>>> x = torch.rand(1,3,8,8, requires_grad=True)
>>> f = lambda z: z.mean(dim=(1,2,3))  # простой скаляр
>>> IG = integrated_gradients(x, f, x0="zeros", m=10)
>>> IG.shape == x.shape
True
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import warnings
import torch

def _make_baseline(x: torch.Tensor, x0: Optional[torch.Tensor | str], seed: int = 0) -> torch.Tensor:
    if isinstance(x0, torch.Tensor):
        return x0.to(x.device).type_as(x)
    if isinstance(x0, str):
        mode = x0.lower()
        if mode == "zeros":
            return torch.zeros_like(x)
        elif mode == "mean":
            m = x.mean(dim=(2,3), keepdim=True)
            return torch.cat([m[:,i:i+1] for i in range(x.shape[1])], dim=1).expand_as(x)
        elif mode == "noise":
            g = torch.Generator(device=x.device)
            g.manual_seed(seed)
            return torch.rand_like(x, generator=g) * 2 - 1  # в [-1,1]
    # default
    return torch.zeros_like(x)


@torch.no_grad()
def _chunked_mean_grads(inputs: torch.Tensor, alphas: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor],
                        x0: torch.Tensor, x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """
    Вычисляет средний градиент по alpha батчами для экономии памяти.
    """
    grads_acc = torch.zeros_like(x)
    n = alphas.numel()
    start = 0
    while start < n:
        end = min(n, start + chunk)
        a = alphas[start:end].view(-1, *([1] * (x.ndim - 1))).to(x.device)
        a.requires_grad_(False)
        xi = x0 + a * (x - x0)
        xi.requires_grad_(True)
        yi = f(xi)
        if yi.ndim > 1:
            yi = yi.view(yi.shape[0], -1).mean(dim=1)
        # суммируем по batch на текущих alphas
        ysum = yi.sum()
        grads = torch.autograd.grad(ysum, xi)[0]
        grads_acc += grads.mean(dim=0)
        start = end
    grads_mean = grads_acc / (n / chunk if n >= chunk else 1.0)
    return grads_mean


def integrated_gradients(x: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor],
                         x0: Optional[torch.Tensor | str] = None, m: int = 20) -> torch.Tensor:
    """
    Вычисляет интегрированные градиенты IG.

    Parameters
    ----------
    x : torch.Tensor
        Вход (B,C,H,W), требуется градиент.
    f : Callable
        Функция скалярной релевантности по batch (возвращает (B,) или (B,1)).
    x0 : torch.Tensor | str | None
        Бейзлайн. Если строка: "mean"|"noise"|"zeros".
        Если None — "zeros".
    m : int
        Количество точек интегрирования.

    Returns
    -------
    torch.Tensor
        IG карта (B,C,H,W).

    Math
    ----
    IG_i(x) = (x_i - x_{0,i}) * mean_{α_k} grad_i f(x_α_k),
    где x_α = x_0 + α(x - x_0), α_k = k/m, k=1..m.

    Completeness
    ------------
    sum_i IG_i ≈ f(x) - f(x_0)
    """
    needs_grad_reset = not x.requires_grad
    x = x.clone().detach().requires_grad_(True)
    x0t = _make_baseline(x, "zeros" if x0 is None else x0)
    alphas = torch.linspace(0.0, 1.0, steps=m, device=x.device)
    # вычисляем средний градиент по alpha батчами
    grads_mean = _chunked_mean_grads(x, alphas, f, x0t, x, chunk=max(8, m))
    IG = (x - x0t) * grads_mean

    # Проверка полноты (мягко, предупреждение)
    with torch.no_grad():
        fx = f(x.detach())
        fx0 = f(x0t.detach())
        if fx.ndim > 1:
            fx = fx.view(fx.shape[0], -1).mean(dim=1)
        if fx0.ndim > 1:
            fx0 = fx0.view(fx0.shape[0], -1).mean(dim=1)
        lhs = IG.view(IG.shape[0], -1).sum(dim=1)
        rhs = fx - fx0
        err = (lhs - rhs).abs().mean().item()
        if err > 0.1:
            warnings.warn(f"IG completeness deviation mean |sum(IG)-(f(x)-f(x0))|={err:.4f} > 0.1")

    if needs_grad_reset:
        IG = IG.detach()
    return IG


def trajectory_ig(x: torch.Tensor,
                  f_t: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  x0_t_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  t_list: list,
                  m: int = 10,
                  agg: str = "mean") -> dict:
    """
    Траекторные IG по шагам диффузии.

    Parameters
    ----------
    x : torch.Tensor
        (B,C,H,W).
    f_t : Callable
        Скалярная релевантность f_t(x_t, t) -> (B,).
    x0_t_fn : Callable
        Бейзлайн по времени: x0_t = x0_t_fn(x, t).
    t_list : list
        Список таймстепов (torch.long или int).
    m : int
        Точек интегрирования.
    agg : str
        "mean"|"median" для агрегированной карты по времени.

    Returns
    -------
    dict
        {t: IG_t (B,C,H,W), "agg": aggregated (B,C,H,W)}
    """
    results = {}
    all_maps = []
    device = x.device
    for t in t_list:
        t_t = torch.tensor([t] * x.shape[0], dtype=torch.long, device=device)
        def f_wrap(xi):
            return f_t(xi, t_t)
        x0t = x0_t_fn(x, t_t)
        IG_t = integrated_gradients(x, f_wrap, x0=x0t, m=m)
        results[int(t)] = IG_t
        all_maps.append(IG_t)
    if agg == "mean":
        agg_map = torch.stack(all_maps, dim=0).mean(dim=0)
    else:
        agg_map = torch.median(torch.stack(all_maps, dim=0), dim=0).values
    results["agg"] = agg_map
    return results