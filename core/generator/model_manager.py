"""
Управление моделями для ISIC Generator
Управление моделями машинного обучения
"""

import os
import torch
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from diffusers import UNet2DModel, DDPMScheduler

from ..cache import CacheManager
from ..utils import PathManager, Logger
from ..config import ConfigManager


class ModelManager:
    """Менеджер для загрузки и управления моделями"""
    
    def __init__(self, config_manager: ConfigManager, cache_manager: CacheManager, 
                 path_manager: PathManager, logger: Logger):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.path_manager = path_manager
        self.logger = logger
        
        # Загруженные модели
        self.loaded_models: Dict[str, UNet2DModel] = {}
        self.loaded_schedulers: Dict[str, DDPMScheduler] = {}
        
        # Метаданные моделей
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Устройство по умолчанию
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_available_classes(self) -> List[str]:
        """Возвращает список доступных классов"""
        try:
            checkpoints_path = self.config_manager.get_path("checkpoints")
            if not checkpoints_path:
                return []
                
            available_classes = []
            for file_path in Path(checkpoints_path).glob("unet_*_best.pth"):
                # Извлекаем название класса из имени файла
                class_name = file_path.stem.replace("unet_", "").replace("_best", "")
                available_classes.append(class_name)
                
            self.logger.log_info(f"Найдено {len(available_classes)} доступных классов: {available_classes}")
            return available_classes
            
        except Exception as e:
            self.logger.log_error(f"Ошибка получения доступных классов: {e}")
            return []
            
    def get_model_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о модели для указанного класса"""
        try:
            checkpoints_path = self.config_manager.get_path("checkpoints")
            if not checkpoints_path:
                return None
                
            model_path = Path(checkpoints_path) / f"unet_{class_name}_best.pth"
            
            if not model_path.exists():
                return None
                
            # Получаем размер файла
            file_size = model_path.stat().st_size
            
            # Проверяем, загружена ли модель
            is_loaded = class_name in self.loaded_models
            
            return {
                "class_name": class_name,
                "model_path": str(model_path),
                "file_size": file_size,
                "is_loaded": is_loaded,
                "device": str(self.device)
            }
            
        except Exception as e:
            self.logger.log_error(f"Ошибка получения информации о модели {class_name}: {e}")
            return None
            
    def load_model(self, class_name: str, force_reload: bool = False) -> bool:
        """Загружает модель для указанного класса"""
        try:
            # Проверяем, загружена ли уже модель
            if class_name in self.loaded_models and not force_reload:
                self.logger.log_info(f"Модель для класса {class_name} уже загружена")
                return True
                
            # Получаем путь к модели
            checkpoints_path = self.config_manager.get_path("checkpoints")
            if not checkpoints_path:
                self.logger.log_error("Путь к чекпоинтам не настроен")
                return False
                
            model_path = Path(checkpoints_path) / f"unet_{class_name}_best.pth"
            
            if not model_path.exists():
                self.logger.log_error(f"Файл модели не найден: {model_path}")
                return False
                
            # Проверяем кэш по имени класса и при наличии грузим веса из кэш-файла
            cached_path = self.cache_manager.get_cached_model(class_name)
            if cached_path and not force_reload:
                self.logger.log_info(f"Загружаем модель {class_name} из кэша: {cached_path}")
                model = self._create_model_architecture()
                checkpoint = torch.load(cached_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                model = model.to(self.device)
                model.eval()
                self.loaded_models[class_name] = model
                # Инициализируем планировщик
                scheduler = self.create_scheduler(class_name)
                self.loaded_schedulers[class_name] = scheduler
                # Метаданные
                import time
                self.model_metadata[class_name] = {
                    "model_path": str(cached_path),
                    "loaded_at": time.time(),
                    "device": str(self.device)
                }
                return True
                
            # Загружаем модель
            self.logger.log_info(f"Загружаем модель для класса {class_name}...")
            
            # Создаем архитектуру модели
            model = self._create_model_architecture()
            
            # Загружаем веса
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            
            # Перемещаем на устройство
            model = model.to(self.device)
            model.eval()
            
            # Сохраняем модель
            self.loaded_models[class_name] = model
            
            # Кэшируем модельный файл-источник (копирует в кэш при отсутствии)
            try:
                self.cache_manager.cache_model(str(model_path), class_name)
            except Exception as e_cache:
                self.logger.log_warning(f"Не удалось закэшировать модель {class_name}: {e_cache}")
            
            # Создаем планировщик
            scheduler = self.create_scheduler(class_name)
            self.loaded_schedulers[class_name] = scheduler
            
            # Сохраняем метаданные
            import time
            self.model_metadata[class_name] = {
                "model_path": str(model_path),
                "loaded_at": time.time(),
                "device": str(self.device)
            }
            
            self.logger.log_info(f"Модель для класса {class_name} успешно загружена")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Ошибка загрузки модели для класса {class_name}: {e}")
            return False
            
    def _create_model_architecture(self) -> UNet2DModel:
        """Создает архитектуру UNet модели согласно train_diffusion.py"""
        return UNet2DModel(
            sample_size=128,  # Размер изображения из train_diffusion.py
            in_channels=3,    # RGB каналы
            out_channels=3,   # RGB каналы
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),  # Архитектура из train_diffusion.py
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D", 
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
            class_embed_type=None,  # Дополнительный параметр из train_diffusion.py
        )
        
    def create_scheduler(self, class_name: str) -> DDPMScheduler:
        """Создает планировщик для указанного класса согласно train_diffusion.py"""
        try:
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,  # TIMESTEPS из train_diffusion.py
                beta_schedule="squaredcos_cap_v2"  # beta_schedule из train_diffusion.py
            )
            # Установим число шагов инференса из конфига (фолбэк на 50)
            try:
                steps = int(self.config_manager.get_generation_param("inference_timesteps"))
            except Exception:
                steps = 50
            steps = max(1, min(1000, steps))
            scheduler.set_timesteps(steps)
            
            self.logger.log_info(f"Создан планировщик для класса {class_name}")
            return scheduler
            
        except Exception as e:
            self.logger.log_error(f"Ошибка создания планировщика для класса {class_name}: {e}")
            # Корректный фолбэк с теми же параметрами
            try:
                fallback = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2",
                    prediction_type="epsilon"
                )
                return fallback
            except Exception:
                # Последний шанс – вернуть дефолтный планировщик, вызывающийся без параметров
                return DDPMScheduler()
            
    def unload_model(self, class_name: str) -> bool:
        """Выгружает модель для указанного класса"""
        try:
            if class_name in self.loaded_models:
                # Очищаем память GPU
                if torch.cuda.is_available():
                    del self.loaded_models[class_name]
                    torch.cuda.empty_cache()
                else:
                    del self.loaded_models[class_name]
                    
                # Удаляем планировщик
                if class_name in self.loaded_schedulers:
                    del self.loaded_schedulers[class_name]
                    
                # Удаляем метаданные
                if class_name in self.model_metadata:
                    del self.model_metadata[class_name]
                    
                self.logger.log_info(f"Модель для класса {class_name} выгружена")
                return True
                
            return False
            
        except Exception as e:
            self.logger.log_error(f"Ошибка выгрузки модели для класса {class_name}: {e}")
            return False
            
    def unload_all_models(self) -> bool:
        """Выгружает все загруженные модели"""
        try:
            success = True
            for class_name in list(self.loaded_models.keys()):
                if not self.unload_model(class_name):
                    success = False
                    
            if success:
                self.logger.log_info("Все модели выгружены")
            else:
                self.logger.log_warning("Некоторые модели не удалось выгрузить")
                
            return success
            
        except Exception as e:
            self.logger.log_error(f"Ошибка выгрузки всех моделей: {e}")
            return False
            
    def validate_model(self, class_name: str) -> bool:
        """Проверяет валидность модели для указанного класса"""
        try:
            if class_name not in self.loaded_models:
                self.logger.log_warning(f"Модель для класса {class_name} не загружена")
                return False
                
            model = self.loaded_models[class_name]
            
            # Проверяем, что модель на правильном устройстве
            try:
                model_device = next(model.parameters()).device
            except Exception:
                model_device = torch.device("cpu")
            if model_device.type != self.device.type or (model_device.type == "cuda" and model_device.index != self.device.index):
                self.logger.log_warning(f"Модель {class_name} находится на неправильном устройстве: {model_device} != {self.device}")
                return False
                
            # Проверяем, что модель в режиме eval
            if model.training:
                self.logger.log_warning(f"Модель {class_name} не в режиме eval")
                return False
                
            self.logger.log_info(f"Модель для класса {class_name} валидна")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Ошибка валидации модели для класса {class_name}: {e}")
            return False
            
    def get_loaded_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает информацию о загруженных моделях"""
        info = {}
        
        for class_name, model in self.loaded_models.items():
            info[class_name] = {
                "device": str(model.device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "metadata": self.model_metadata.get(class_name, {})
            }
            
        return info
        
    def change_device(self, new_device: str) -> bool:
        """Изменяет устройство для всех загруженных моделей"""
        try:
            device = torch.device(new_device)
            
            # Проверяем доступность устройства
            if device.type == "cuda" and not torch.cuda.is_available():
                self.logger.log_error("CUDA недоступна")
                return False
                
            self.device = device
            
            # Перемещаем все модели на новое устройство
            for class_name, model in self.loaded_models.items():
                try:
                    self.loaded_models[class_name] = model.to(device)
                    self.logger.log_info(f"Модель {class_name} перемещена на {device}")
                except Exception as e:
                    self.logger.log_error(f"Ошибка перемещения модели {class_name}: {e}")
                    return False
                    
            self.logger.log_info(f"Все модели перемещены на {device}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Ошибка изменения устройства: {e}")
            return False
            
    def cleanup(self):
        """Очищает ресурсы"""
        try:
            self.unload_all_models()
            self.logger.log_info("ModelManager очищен")
        except Exception as e:
            self.logger.log_error(f"Ошибка очистки ModelManager: {e}")
