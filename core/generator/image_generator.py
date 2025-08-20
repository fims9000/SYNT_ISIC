"""
Image generation for ISIC Generator
Генерация изображений с использованием обученных моделей
"""

import os
import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from ..config import ConfigManager
from ..utils import PathManager, Logger
from ..cache import CacheManager
from .model_manager import ModelManager


class ImageGenerator:
    """Основной класс для генерации изображений"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        # Инициализируем компоненты
        self.path_manager = PathManager()
        self.logger = Logger()
        self.cache_manager = CacheManager()
        self.model_manager = ModelManager(
            config_manager, 
            self.cache_manager, 
            self.path_manager, 
            self.logger
        )
        
        # Callback'и для обновления прогресса и логирования
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        self.log_callback: Optional[Callable[[str], None]] = None
        
        # Состояние генерации
        self.is_generating = False
        self.stop_requested = False
        
        # Загруженные цветовые статистики
        self.color_statistics: Dict[str, Dict[str, Any]] = {}
        
        # Устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Кэш для доступных классов
        self._cached_available_classes = None
        
        # Загружаем цветовые статистики
        self._load_color_statistics()
        
        # Предзагружаем все доступные модели (после полной инициализации)
        try:
            self._preload_all_models()
        except Exception as e:
            # Если предзагрузка не удалась, логируем ошибку, но не прерываем инициализацию
            if hasattr(self, '_log_message'):
                self._log_message(f"Предзагрузка моделей не удалась: {e}", "warning")
            else:
                print(f"Предзагрузка моделей не удалась: {e}")
            
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Устанавливает callback для обновления прогресса"""
        self.progress_callback = callback
        
    def set_log_callback(self, callback: Callable[[str], None]):
        """Устанавливает callback для логирования"""
        self.log_callback = callback
        
    def _update_progress(self, current: int, total: int, message: str):
        """Обновляет прогресс"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
            
    def _log_message(self, message: str, level: str = "info"):
        """Логирует сообщение"""
        if self.log_callback:
            self.log_callback(f"[{level.upper()}] {message}")
            
        # Также логируем в основной логгер
        if level == "error":
            self.logger.log_error(message)
        elif level == "warning":
            self.logger.log_warning(message)
        else:
            self.logger.log_info(message)
            
    def _load_color_statistics(self):
        """Загружает цветовые статистики из файла"""
        try:
            checkpoints_path = self.config_manager.get_path("checkpoints")
            if not checkpoints_path:
                self._log_message("Путь к чекпоинтам не настроен, цветовые статистики не загружены", "warning")
                return
                
            color_stats_path = Path(checkpoints_path) / "color_statistics.json"
            
            if color_stats_path.exists():
                with open(color_stats_path, 'r') as f:
                    self.color_statistics = json.load(f)
                    
                if hasattr(self, '_log_message'):
                    self._log_message(f"Загружены цветовые статистики для {len(self.color_statistics)} классов")
                else:
                    print(f"Загружены цветовые статистики для {len(self.color_statistics)} классов")
            else:
                if hasattr(self, '_log_message'):
                    self._log_message("Файл color_statistics.json не найден", "warning")
                else:
                    print("Файл color_statistics.json не найден")
                
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Ошибка загрузки цветовых статистик: {e}", "error")
            else:
                print(f"Ошибка загрузки цветовых статистик: {e}")
            
    def _preload_all_models(self):
        """Предзагружает все доступные модели один раз"""
        try:
            # Проверяем, что метод логирования доступен
            if not hasattr(self, '_log_message'):
                print("Метод логирования недоступен, пропускаем предзагрузку")
                return
                
            # Проверяем, что model_manager доступен
            if not hasattr(self, 'model_manager') or not self.model_manager:
                print("Model manager недоступен, пропускаем предзагрузку")
                return
                
            available_classes = self.get_available_classes()
            if not available_classes:
                print("Нет доступных классов для предзагрузки")
                return
                
            if hasattr(self, '_log_message'):
                self._log_message(f"Предзагружаю {len(available_classes)} моделей...")
            else:
                print(f"Предзагружаю {len(available_classes)} моделей...")
            
            for class_name in available_classes:
                try:
                    success = self.model_manager.load_model(class_name)
                    if success:
                        if hasattr(self, '_log_message'):
                            self._log_message(f"Модель {class_name} предзагружена успешно")
                        else:
                            print(f"Модель {class_name} предзагружена успешно")
                    else:
                        if hasattr(self, '_log_message'):
                            self._log_message(f"Не удалось предзагрузить модель {class_name}", "warning")
                        else:
                            print(f"Не удалось предзагрузить модель {class_name}")
                except Exception as e:
                    if hasattr(self, '_log_message'):
                        self._log_message(f"Ошибка предзагрузки модели {class_name}: {e}", "error")
                    else:
                        print(f"Ошибка предзагрузки модели {class_name}: {e}")
                    
            if hasattr(self, '_log_message'):
                self._log_message("Предзагрузка моделей завершена")
            else:
                print("Предзагрузка моделей завершена")
            
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Критическая ошибка предзагрузки моделей: {e}", "error")
            else:
                print(f"Критическая ошибка предзагрузки моделей: {e}")
            
    def get_available_classes(self) -> List[str]:
        """Возвращает список доступных классов (с кэшированием)"""
        try:
            # Возвращаем кэшированный результат если есть
            if self._cached_available_classes is not None:
                return self._cached_available_classes
                
            # Получаем и кэшируем список классов
            if hasattr(self, 'model_manager') and self.model_manager:
                self._cached_available_classes = self.model_manager.get_available_classes()
                return self._cached_available_classes
            else:
                return []
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Ошибка получения доступных классов: {e}", "error")
            else:
                print(f"Ошибка получения доступных классов: {e}")
            return []
        
    def validate_models(self, class_names: List[str]) -> Dict[str, bool]:
        """Проверяет валидность моделей для указанных классов"""
        results = {}
        
        try:
            if not hasattr(self, 'model_manager') or not self.model_manager:
                for class_name in class_names:
                    results[class_name] = False
                return results
                
            for class_name in class_names:
                try:
                    # Загружаем модель если не загружена
                    if class_name not in self.model_manager.loaded_models:
                        success = self.model_manager.load_model(class_name)
                        if not success:
                            results[class_name] = False
                            continue
                            
                    # Проверяем валидность
                    is_valid = self.model_manager.validate_model(class_name)
                    results[class_name] = is_valid
                    
                except Exception as e:
                    if hasattr(self, '_log_message'):
                        self._log_message(f"Ошибка валидации модели {class_name}: {e}", "error")
                    else:
                        print(f"Ошибка валидации модели {class_name}: {e}")
                    results[class_name] = False
                    
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Критическая ошибка валидации моделей: {e}", "error")
            else:
                print(f"Критическая ошибка валидации моделей: {e}")
            for class_name in class_names:
                results[class_name] = False
                
        return results
        
    def _create_model(self, class_name: str) -> UNet2DModel:
        """Создает модель с правильной архитектурой для чекпоинтов"""
        model = UNet2DModel(
            sample_size=128,  # Правильный размер для чекпоинтов
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
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
            class_embed_type=None,
        )
        return model
        
    def _create_scheduler(self) -> DDPMScheduler:
        """Создает планировщик с правильными параметрами"""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        scheduler.set_timesteps(1000)
        return scheduler
        
    def _cleanup_memory(self):
        """Очищает память CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def generate_single_image(self, class_name: str, output_path: str, 
                            postprocess: bool = True) -> bool:
        """Генерирует одно изображение для указанного класса"""
        try:
            if self.stop_requested:
                return False
                
            # Проверяем, что model_manager доступен
            if not hasattr(self, 'model_manager') or not self.model_manager:
                if hasattr(self, '_log_message'):
                    self._log_message("Model manager недоступен", "error")
                else:
                    print("Model manager недоступен")
                return False
                
            # Очищаем память перед генерацией
            self._cleanup_memory()
                
            # Проверяем, загружена ли модель
            if class_name not in self.model_manager.loaded_models:
                success = self.model_manager.load_model(class_name)
                if not success:
                    if hasattr(self, '_log_message'):
                        self._log_message(f"Не удалось загрузить модель для класса {class_name}", "error")
                    else:
                        print(f"Не удалось загрузить модель для класса {class_name}")
                    return False
                    
            # Получаем модель и планировщик
            model = self.model_manager.loaded_models[class_name]
            scheduler = self._create_scheduler()
            
            # Генерируем изображение
            with torch.no_grad():
                # Создаем случайный шум с правильным размером
                noise = torch.randn(1, 3, 128, 128, device=self.device)
                
                # Процесс денойзинга
                latents = noise
                for t in scheduler.timesteps:
                    if self.stop_requested:
                        return False
                        
                    # Предсказываем шум
                    noise_pred = model(latents, t).sample
                    
                    # Обновляем латент
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    
            # Конвертируем в изображение
            image = latents.squeeze(0).permute(1, 2, 0)
            image = (image + 1) / 2  # Нормализуем в [0, 1]
            image = torch.clamp(image, 0, 1)
            
            # Конвертируем в PIL
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Применяем постобработку цветов
            if postprocess:
                pil_image = self._apply_color_postprocessing(pil_image, class_name)
                
            # Сохраняем изображение
            pil_image.save(output_path)
            
            # Очищаем память после генерации
            self._cleanup_memory()
            
            if hasattr(self, '_log_message'):
                self._log_message(f"Изображение для класса {class_name} сгенерировано: {output_path}")
            else:
                print(f"Изображение для класса {class_name} сгенерировано: {output_path}")
            return True
            
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Ошибка генерации изображения для класса {class_name}: {e}", "error")
            else:
                print(f"Ошибка генерации изображения для класса {class_name}: {e}")
            self._cleanup_memory()
            return False
            
    def _apply_color_postprocessing(self, image: Image.Image, class_name: str) -> Image.Image:
        """Применяет постобработку цветов на основе статистик"""
        try:
            if class_name not in self.color_statistics:
                return image
                
            stats = self.color_statistics[class_name]
            
            # Конвертируем в numpy
            img_array = np.array(image)
            
            # Получаем статистики RGB
            if "rgb" in stats and "mean" in stats["rgb"]:
                target_mean = np.array(stats["rgb"]["mean"])
                target_std = np.array(stats["rgb"]["std"])
                
                # Нормализуем текущее изображение
                current_mean = np.mean(img_array, axis=(0, 1))
                current_std = np.std(img_array, axis=(0, 1))
                
                # Применяем нормализацию
                img_array = ((img_array - current_mean) / current_std) * target_std + target_mean
                
                # Ограничиваем значения
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
            # Конвертируем обратно в PIL
            return Image.fromarray(img_array)
            
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Ошибка постобработки цветов для класса {class_name}: {e}", "warning")
            else:
                print(f"Ошибка постобработки цветов для класса {class_name}: {e}")
            return image
            
    def generate_images(self, class_configs: List[Tuple[str, int]], 
                       output_dir: str, postprocess: bool = True) -> Dict[str, Any]:
        """Генерирует изображения для указанных классов с постоянной очисткой памяти"""
        try:
            if self.is_generating:
                if hasattr(self, '_log_message'):
                    self._log_message("Генерация уже запущена", "warning")
                else:
                    print("Генерация уже запущена")
                return {"error": "Генерация уже запущена"}
                
            # Проверяем, что model_manager доступен
            if not hasattr(self, 'model_manager') or not self.model_manager:
                if hasattr(self, '_log_message'):
                    self._log_message("Model manager недоступен", "error")
                else:
                    print("Model manager недоступен")
                return {"error": "Model manager недоступен"}
                
            self.is_generating = True
            self.stop_requested = False
            
            # Создаем выходную директорию
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Инициализируем CSV для метаданных
            csv_path = output_path / "synthetic_dataset.csv"
            self._initialize_synthetic_csv(csv_path)
            
            total_images = sum(count for _, count in class_configs)
            generated_count = 0
            
            if hasattr(self, '_log_message'):
                self._log_message(f"Начинаем генерацию {total_images} изображений")
            else:
                print(f"Начинаем генерацию {total_images} изображений")
            
            # Генерируем изображения для каждого класса
            for class_name, count in class_configs:
                if self.stop_requested:
                    break
                    
                if hasattr(self, '_log_message'):
                    self._log_message(f"Generating {count} images for class {class_name}")
                else:
                    print(f"Generating {count} images for class {class_name}")
                
                # Создаем папку для класса
                class_dir = output_path / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Генерируем изображения по одному с постоянной очисткой памяти
                for i in range(count):
                    if self.stop_requested:
                        break
                        
                    # Генерируем имя файла в формате ISIC
                    isic_number = self.path_manager.get_next_isic_number(str(output_path))
                    filename = self.path_manager.get_isic_filename(isic_number)
                    file_path = class_dir / f"{filename}.png"
                    
                    # Генерируем изображение
                    success = self.generate_single_image(class_name, str(file_path), postprocess)
                    
                    if success:
                        generated_count += 1
                        
                        # Добавляем в CSV
                        self._append_to_csv(csv_path, {
                            "filename": f"{filename}.png",
                            "class": class_name,
                            "isic_number": isic_number,
                            "source": "synthetic",
                            "generated_at": str(Path(file_path).stat().st_mtime)
                        })
                        
                        # Обновляем прогресс
                        self._update_progress(generated_count, total_images, 
                                           f"Generated {generated_count}/{total_images}")
                        
                        # Очищаем память после КАЖДОГО изображения
                        self._cleanup_memory()
                        
                        # Логируем использование памяти
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                            if hasattr(self, '_log_message'):
                                self._log_message(f"Memory after image {generated_count}: {memory_used:.3f}GB")
                            else:
                                print(f"Memory after image {generated_count}: {memory_used:.3f}GB")
                    
            # Финальное обновление
            if self.stop_requested:
                if hasattr(self, '_log_message'):
                    self._log_message("Generation stopped by user")
                else:
                    print("Generation stopped by user")
                result = {"total_generated": generated_count, "stopped": True}
            else:
                if hasattr(self, '_log_message'):
                    self._log_message(f"Generation completed. Generated {generated_count} images")
                else:
                    print(f"Generation completed. Generated {generated_count} images")
                result = {"total_generated": generated_count, "stopped": False}
                
            self.is_generating = False
            return result
            
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Critical generation error: {e}", "error")
            else:
                print(f"Critical generation error: {e}")
            self.is_generating = False
            return {"error": str(e)}
            
    def _initialize_synthetic_csv(self, csv_path: Path):
        """Инициализирует CSV файл для метаданных"""
        try:
            headers = ["filename", "class", "isic_number", "source", "generated_at"]
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
            if hasattr(self, '_log_message'):
                self._log_message(f"CSV metadata file created: {csv_path}")
            else:
                print(f"CSV metadata file created: {csv_path}")
                
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Error creating CSV file: {e}", "error")
            else:
                print(f"Error creating CSV file: {e}")
            
    def _append_to_csv(self, csv_path: Path, data: Dict[str, str]):
        """Добавляет запись в CSV файл"""
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writerow(data)
                
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Error writing to CSV: {e}", "error")
            else:
                print(f"Error writing to CSV: {e}")
            
    def stop_generation(self):
        """Останавливает генерацию"""
        self.stop_requested = True
        if hasattr(self, '_log_message'):
            self._log_message("Generation stop requested")
        else:
            print("Generation stop requested")
        
    def get_generation_status(self) -> Dict[str, Any]:
        """Возвращает статус генерации"""
        try:
            if hasattr(self, 'model_manager') and self.model_manager:
                loaded_models = list(self.model_manager.loaded_models.keys())
            else:
                loaded_models = []
                
            return {
                "is_generating": self.is_generating,
                "stop_requested": self.stop_requested,
                "loaded_models": loaded_models,
                "device": str(self.device)
            }
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Ошибка получения статуса генерации: {e}", "error")
            else:
                print(f"Ошибка получения статуса генерации: {e}")
            return {
                "is_generating": False,
                "stop_requested": False,
                "loaded_models": [],
                "device": "unknown"
            }
        
    def cleanup(self):
        """Очищает ресурсы"""
        try:
            self.stop_generation()
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.cleanup()
            if hasattr(self, 'cache_manager') and self.cache_manager:
                self.cache_manager.cleanup_temp_files()
            self._cleanup_memory()
            if hasattr(self, '_log_message'):
                self._log_message("ImageGenerator cleaned up")
            else:
                print("ImageGenerator cleaned up")
        except Exception as e:
            if hasattr(self, '_log_message'):
                self._log_message(f"Error cleaning up ImageGenerator: {e}", "error")
            else:
                print(f"Error cleaning up ImageGenerator: {e}")
