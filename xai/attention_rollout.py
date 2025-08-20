"""
attention_rollout.py

Attention Rollout агрегирует матрицы внимания по слоям.

Математика:
- Эффективное внимание:
  A_eff = ∏_l A^(l)
  где A^(l) — матрица внимания слоя l.
  Часто используют добавление I (skip) и нормировку.

Опции:
- add_identity: A^(l) := (A^(l) + I)/2
- normalize: "row"|"col"|"none"
- head_agg: "mean"|"max" — как агрегировать по головам.

Doctest:
>>> import numpy as np
>>> A1 = np.eye(4)[None,None,:,:]
>>> A2 = np.eye(4)[None,None,:,:]
>>> A_eff = attention_rollout([A1, A2])
>>> np.allclose(A_eff, np.eye(4))
True
"""
from __future__ import annotations
from typing import List
import numpy as np

def _normalize(A: np.ndarray, how: str) -> np.ndarray:
    if how == "row":
        s = A.sum(axis=-1, keepdims=True) + 1e-8
        return A / s
    if how == "col":
        s = A.sum(axis=-2, keepdims=True) + 1e-8
        return A / s
    return A

def attention_rollout(attn_stack: List[np.ndarray],
                      add_identity: bool = True,
                      normalize: str = "row",
                      head_agg: str = "mean") -> np.ndarray:
    """
    Parameters
    ----------
    attn_stack : list of np.ndarray
        Список A^(l) формы (B, H, N, N) либо (H,N,N) либо (N,N).
    add_identity : bool
        Добавлять I перед нормировкой.
    normalize : str
        "row"|"col"|"none".
    head_agg : str
        "mean"|"max".

    Returns
    -------
    np.ndarray
        A_eff (B,N,N) либо (N,N) если B=1.

    Notes
    -----
    Численная стабильность обеспечивается нормировкой и малой eps.
    """
    mats = []
    B = None
    for A in attn_stack:
        A = np.asarray(A, dtype=np.float32)
        if A.ndim == 2:
            A = A[None, None, ...]
        elif A.ndim == 3:
            A = A[None, ...]
        elif A.ndim != 4:
            raise ValueError("A must be (B,H,N,N)/(H,N,N)/(N,N)")
        if B is None:
            B = A.shape[0]
        if add_identity:
            N = A.shape[-1]
            I = np.eye(N, dtype=np.float32)
            A = (A + I[None, None]) / 2.0
        # agg heads
        if head_agg == "mean":
            A = A.mean(axis=1)
        else:
            A = A.max(axis=1)
        A = _normalize(A, normalize)  # (B,N,N)
        mats.append(A)
    # product
    A_eff = mats[0]
    for k in range(1, len(mats)):
        A_eff = A_eff @ mats[k]
    if A_eff.shape[0] == 1:
        return A_eff[0]
    return A_eff