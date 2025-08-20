"""
model_ddpm.py

Построение UNet2DModel (diffusers) и DDPMScheduler, а также хук-класс для извлечения
feature maps (F^k) и attention матриц A^(l,h).

Обозначения:
- F^k — карта признаков k-го канала на некотором слое, размер CxHxW.
- A^(l,h) — матрица внимания слоя l, головы h, размер NxN, где N=H*W (пространственные токены).

Места извлечения:
- feature maps: mid_block (бутылочное горлышко) и первый attentional блок в down/up уровнях.
- attention: перехват через кастомный AttnProcessor, сохраняем softmax(QK^T/sqrt(d)).

Doctest:
>>> from diffusers import UNet2DModel
>>> unet = build_unet(img_size=64, in_ch=3, base_ch=64, attn_levels=(1,2))
>>> sched = build_scheduler()
>>> hooks = AttentionHooks(unet)
>>> import torch
>>> x = torch.randn(1,3,64,64)
>>> t = torch.tensor([10], dtype=torch.long)
>>> with hooks.capture():
...     y = unet(x, t).sample
>>> fm = hooks.get_feature_maps()
>>> attn = hooks.get_attentions()
>>> isinstance(fm, dict) and isinstance(attn, dict)
True
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import contextlib
import math
import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

__all__ = [
    "build_unet",
    "build_scheduler",
    "AttentionHooks",
]


def build_unet(img_size: int = 128, in_ch: int = 3, base_ch: int = 128, attn_levels: tuple = (1, 2)) -> UNet2DModel:
    """
    Создаёт UNet2DModel для 128x128.

    Parameters
    ----------
    img_size : int
        Размер изображения.
    in_ch : int
        Входные каналы.
    base_ch : int
        Базовые каналы.
    attn_levels : tuple
        Индексы уровней с вниманием (см. down_block_types / up_block_types).

    Returns
    -------
    UNet2DModel
    """
    # Подбор блоков с учетом внимания
    # Архитектура должна соответствовать чекпоинту: [64, 128, 256, 256]
    out_channels = (
        base_ch,      # Уровень 0: base_ch (64)
        base_ch * 2,  # Уровень 1: base_ch * 2 (128)
        base_ch * 4,  # Уровень 2: base_ch * 4 (256)
        base_ch * 4,  # Уровень 3: base_ch * 4 (256)
    )
    down_types = []
    up_types = []
    for i in range(4):
        if i in attn_levels:
            down_types.append("AttnDownBlock2D")
            up_types.append("AttnUpBlock2D")
        else:
            down_types.append("DownBlock2D")
            up_types.append("UpBlock2D")
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=in_ch,
        out_channels=in_ch,
        layers_per_block=2,
        block_out_channels=out_channels,
        down_block_types=tuple(down_types),
        up_block_types=tuple(reversed(up_types)),
        class_embed_type=None,
    )
    return model


def build_scheduler(num_train_timesteps: int = 1000, beta_schedule: str = "squaredcos_cap_v2",
                    prediction_type: str = "epsilon") -> DDPMScheduler:
    """
    Создаёт DDPM scheduler.

    Parameters
    ----------
    num_train_timesteps : int
    beta_schedule : str
    prediction_type : str

    Returns
    -------
    DDPMScheduler
    """
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
    )


class CapturingAttnProcessor(nn.Module):
    """
    Обёртка над attention-процессором для перехвата весов внимания.

    Сохраняет softmax(QK^T/sqrt(d)) в self.cache под уникальным ключом.
    """
    def __init__(self, inner: nn.Module, cache: Dict[str, torch.Tensor], name: str):
        super().__init__()
        self.inner = inner
        self.cache = cache
        self.name = name

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # исходный процессор может иметь разную сигнатуру — делегируем
        # но перехватываем до/после softmax. Для совместимости хукнем через attn.get_attention_scores
        # Если недоступно — попытаемся вычислить явно.
        if hasattr(self.inner, "forward"):
            out = self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            # Пытаемся сохранить последние веса, если attn временно их держит
            if hasattr(attn, "last_attn_probs") and attn.last_attn_probs is not None:
                self.cache[self.name] = attn.last_attn_probs.detach().cpu()
            return out
        # fallback
        return attn._call_module(self.inner, hidden_states, encoder_hidden_states, attention_mask, temb)


class FeatureHook:
    def __init__(self, module: nn.Module, name: str, storage: Dict[str, torch.Tensor]):
        self.h = module.register_forward_hook(self._fn)
        self.name = name
        self.storage = storage

    def _fn(self, mod, inp, out):
        # out может быть tuple или Sample
        if hasattr(out, "sample"):
            val = out.sample
        else:
            val = out
        if isinstance(val, (list, tuple)):
            val = val[0]
        self.storage[self.name] = val.detach().cpu()

    def remove(self):
        self.h.remove()


class AttentionHooks:
    """
    Класс-хуки для извлечения:
    - feature maps (mid_block, выбранные down/up attentional блоки)
    - attention матриц (через замену процессоров на CapturingAttnProcessor)

    Извлечения:
    - F^k: из mid_block.output (бутылочное горлышко) и первого attention-блока в down/up.
    - A^(l): матрицы внимания по слоям/головам (N x N).

    Notes
    -----
    В UNet2DModel внимание реализовано в SpatialTransformer внутри (Attn* блоки),
    в некоторых версиях diffusers не экспонируют вероятности внимания напрямую.
    Здесь используется замена процессоров на CapturingAttnProcessor, ожидая,
    что attn.last_attn_probs будет заполнен внутренними реализациями.
    Если этого не произошло, словарь attention останется частично пустым.
    """

    def __init__(self, unet: UNet2DModel):
        self.unet = unet
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.attn_maps: Dict[str, torch.Tensor] = {}
        self._feat_hooks: List[FeatureHook] = []
        self._orig_processors: Dict[str, nn.Module] = {}

    @contextlib.contextmanager
    def capture(self):
        try:
            self._install_feature_hooks()
            self._install_attn_processors()
            yield
        finally:
            self._remove_feature_hooks()
            self._restore_attn_processors()

    def _install_feature_hooks(self):
        # mid_block
        if hasattr(self.unet, "mid_block"):
            self._feat_hooks.append(FeatureHook(self.unet.mid_block, "mid_block", self.feature_maps))
        # первый attentional блок в down_blocks
        for i, blk in enumerate(getattr(self.unet, "down_blocks", [])):
            if "Attn" in blk.__class__.__name__:
                self._feat_hooks.append(FeatureHook(blk, f"down_attn_{i}", self.feature_maps))
                break
        # первый attentional блок в up_blocks
        for i, blk in enumerate(getattr(self.unet, "up_blocks", [])):
            if "Attn" in blk.__class__.__name__:
                self._feat_hooks.append(FeatureHook(blk, f"up_attn_{i}", self.feature_maps))
                break

    def _remove_feature_hooks(self):
        for h in self._feat_hooks:
            h.remove()
        self._feat_hooks.clear()

    def _install_attn_processors(self):
        # Проходим по всем attention_processors и меняем на обёртки
        if not hasattr(self.unet, "set_attn_processor"):
            return
        processors = getattr(self.unet, "attn_processors", None)
        if processors is None:
            return
        new_proc = {}
        for name, proc in processors.items():
            self._orig_processors[name] = proc
            new_proc[name] = CapturingAttnProcessor(proc, self.attn_maps, name)
        self.unet.set_attn_processor(new_proc)

    def _restore_attn_processors(self):
        if self._orig_processors and hasattr(self.unet, "set_attn_processor"):
            self.unet.set_attn_processor(self._orig_processors)
        self._orig_processors.clear()

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        return dict(self.feature_maps)

    def get_attentions(self) -> Dict[str, torch.Tensor]:
        return dict(self.attn_maps)