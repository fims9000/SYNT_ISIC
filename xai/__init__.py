#!/usr/bin/env python3
"""
XAI (Explainable AI) пакет для анализа DDPM моделей

Этот пакет предоставляет инструменты для:
- Атрибуции (SHAP, Integrated Gradients, Grad-CAM)
- Attention Rollout
- Контрафактуальных объяснений
- Оценки качества объяснений
"""

__version__ = "1.0.0"
__author__ = "XAI Team"

# Основные модули
from . import attribution_shap
from . import attribution_ig
from . import attribution_gradcam
from . import attention_rollout
from . import counterfactuals
from . import evaluation
from . import latent_viz
from . import model_ddpm
from . import train_or_load
from . import utils

__all__ = [
    'attribution_shap',
    'attribution_ig', 
    'attribution_gradcam',
    'attention_rollout',
    'counterfactuals',
    'evaluation',
    'latent_viz',
    'model_ddpm',
    'train_or_load',
    'utils'
]

