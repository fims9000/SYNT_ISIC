"""
train_or_load.py

Загрузка или инициализация UNet и Scheduler. Опциональный минимальный дообучающий
шаг для совместимости данных (например, один проход MSE по шуму).

Doctest:
>>> from .model_ddpm import build_unet, build_scheduler
>>> unet = load_or_init_unet(None, img_size=32, base_ch=32)
>>> sched = load_scheduler()
>>> isinstance(sched.num_train_timesteps, int)
True
"""
from __future__ import annotations
from typing import Optional
import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
from .model_ddpm import build_unet, build_scheduler

def load_or_init_unet(checkpoint: Optional[str] = None, img_size: int = 128, in_ch: int = 3,
                      base_ch: int = 128, attn_levels: tuple = (1,2)) -> UNet2DModel:
    """
    Грузит чекпоинт или создает новую модель.

    Parameters
    ----------
    checkpoint : str|None
        Путь к .pth (state_dict) или None.
    img_size, in_ch, base_ch, attn_levels
        Параметры архитектуры (должны совпадать с обучением).

    Returns
    -------
    UNet2DModel
    """
    unet = build_unet(img_size=img_size, in_ch=in_ch, base_ch=base_ch, attn_levels=attn_levels)
    if checkpoint is not None and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        unet.load_state_dict(state, strict=False)
    return unet


def load_scheduler(num_train_timesteps: int = 1000,
                   beta_schedule: str = "squaredcos_cap_v2",
                   prediction_type: str = "epsilon") -> DDPMScheduler:
    """
    Возвращает DDPMScheduler.
    """
    return build_scheduler(num_train_timesteps=num_train_timesteps,
                           beta_schedule=beta_schedule,
                           prediction_type=prediction_type)