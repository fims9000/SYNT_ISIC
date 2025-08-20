"""
Core package for ISIC Synthetic Data Generator
Содержит все основные компоненты для генерации изображений
"""

__version__ = "1.0.0"
__author__ = "ISIC Generator Team"

from .config import ConfigManager
from .generator import ImageGenerator
from .utils import Logger, PathManager
from .cache import CacheManager

__all__ = [
    'ConfigManager',
    'ImageGenerator', 
    'Logger',
    'PathManager',
    'CacheManager'
]
