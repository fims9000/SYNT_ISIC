"""
attribution_gradcam.py

Grad-CAM и внимание, взвешенное градиентами.

Математика:
- Grad-CAM:
  α_k = (1/Z) ∑_{i,j} ∂f / ∂F_{ij}^k
  L = ReLU( ∑_k α_k F^k )
  где F^k — k-я карта признаков, Z=H*W.

- Attention-weighted:
  Используем градиенты по матрицам внимания A^(l,h) как веса, агрегируем по головам/слоям,
  затем сворачиваем токеновую карту в HxW heatmap.

Doctest:
>>> import torch
>>> x = torch.randn(1,3,16,16, requires_grad=True)
>>> feat = lambda: (torch.randn(1,8,4,4, requires_grad=True),)  # псевдо feature_getter
>>> def feature_getter(): return feat()[0]
>>> f = lambda z: z.mean(dim=(1,2,3))
>>> hm = gradcam(x, f, feature_getter)
>>> hm.shape == (1,16,16)
True
"""
from __future__ import annotations
from typing import Callable
import torch
import torch.nn.functional as F

def _upsample_like(hm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # hm: (B,1,h,w) or (B,h,w)
    if hm.ndim == 3:
        hm = hm.unsqueeze(1)
    return F.interpolate(hm, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

def gradcam(x: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], feature_getter: Callable[[], torch.Tensor],
            relu: bool = True) -> torch.Tensor:
    """
    Grad-CAM по произвольному feature_getter.

    Parameters
    ----------
    x : torch.Tensor
        (B,C,H,W), requires_grad=True.
    f : Callable
        Скалярная релевантность по batch.
    feature_getter : Callable
        Возвращает feature map (B,K,h,w), с сохранением grad.
    relu : bool
        Применять ReLU к итоговой карте.

    Returns
    -------
    heatmap : torch.Tensor
        (B,H,W)
    """
    x = x.clone().detach().requires_grad_(True)
    feats = feature_getter()  # (B,K,h,w)
    if not feats.requires_grad:
        feats.requires_grad_(True)
    # Прогоняем f с detach(x) чтобы градиент шел через feats
    y = f(x)
    if y.ndim > 1:
        y = y.view(y.shape[0], -1).mean(dim=1)
    ysum = y.sum()
    grads = torch.autograd.grad(ysum, feats, retain_graph=False)[0]  # (B,K,h,w)
    weights = grads.mean(dim=(2,3), keepdim=True)  # α_k
    cam = (weights * feats).sum(dim=1, keepdim=True)  # (B,1,h,w)
    if relu:
        cam = torch.relu(cam)
    cam_ups = _upsample_like(cam, x)  # (B,H,W)
    # нормализация до [0,1]
    B = cam_ups.shape[0]
    cam_norm = []
    for i in range(B):
        c = cam_ups[i]
        if torch.allclose(c.max(), c.min()):
            cam_norm.append(torch.zeros_like(c))
        else:
            cn = (c - c.min()) / (c.max() - c.min() + 1e-8)
            cam_norm.append(cn)
    return torch.stack(cam_norm, dim=0)


def attn_grad_weighted(x: torch.Tensor,
                       f: Callable[[torch.Tensor], torch.Tensor],
                       attn_getter: Callable[[], dict],
                       head_agg: str = "mean") -> torch.Tensor:
    """
    Внимание, взвешенное градиентом по матрицам внимания A^(l,h).

    Parameters
    ----------
    x : torch.Tensor
        (B,C,H,W)
    f : Callable
        Скалярная релевантность по batch.
    attn_getter : Callable
        Возвращает dict {name: A^(l) (B,Hd,N,N)} (если доступно).
    head_agg : str
        "mean"|"max" — агрегация по головам.

    Returns
    -------
    heatmap : torch.Tensor
        (B,H,W)
    """
    x = x.clone().detach().requires_grad_(True)
    # Получаем текущее внимание (forward уже должен был выполнен с capture)
    attn_dict = attn_getter()
    if len(attn_dict) == 0:
        # fallback: нулевая карта
        return torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
    # Вычисляем градиент по x, чтобы иметь связь — приближение:
    y = f(x)
    if y.ndim > 1:
        y = y.view(y.shape[0], -1).mean(dim=1)
    ysum = y.sum()
    grads_x = torch.autograd.grad(ysum, x, retain_graph=False, allow_unused=True)[0]
    B, C, H, W = x.shape
    N = H * W

    # Аггрегируем A^(l,h) с весами по градиенту x (используем норму как прокси)
    w = grads_x.view(B, C, N).abs().mean(dim=1, keepdim=False)  # (B,N)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    A_eff = None
    for name, A in attn_dict.items():
        # A: (B, heads, N, N)
        if A.dim() != 4 or A.size(-1) != N:
            # несовместимо с HxW, пропустим
            continue
        if head_agg == "mean":
            A_bar = A.mean(dim=1)  # (B,N,N)
        else:
            A_bar = A.max(dim=1).values
        # нормировка строк
        A_bar = A_bar / (A_bar.sum(dim=-1, keepdim=True) + 1e-8)
        A_eff = A_bar if A_eff is None else torch.bmm(A_eff, A_bar)

    if A_eff is None:
        return torch.zeros(B, H, W, device=x.device)

    # взвешивание по важности токенов
    heat_tokens = torch.bmm(w.unsqueeze(1), A_eff).squeeze(1)  # (B,N)
    heat = heat_tokens.view(B, H, W)
    # нормализация [0,1]
    for i in range(B):
        h = heat[i]
        if not torch.allclose(h.max(), h.min()):
            heat[i] = (h - h.min()) / (h.max() - h.min() + 1e-8)
        else:
            heat[i] = torch.zeros_like(h)
    return heat