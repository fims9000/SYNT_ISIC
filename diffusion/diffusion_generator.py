#!/usr/bin/env python3
"""
Основной генератор диффузионных изображений с интеграцией постобработки
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from postprocessing.color_postprocessor import ColorPostprocessor
from utils.logging_utils import setup_logger


class DiffusionGenerator:
    """Генератор диффузионных изображений с интегрированной постобработкой"""
    
    def __init__(self, checkpoint_dir: str, stats_path: str, device: str = "cuda:0"):
        """
        Инициализация генератора
        
        Args:
            checkpoint_dir: Директория с чекпоинтами моделей
            stats_path: Путь к файлу с цветовыми статистиками
            device: Устройство для генерации
        """
        self.checkpoint_dir = checkpoint_dir
        self.stats_path = stats_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Инициализируем постпроцессор
        self.postprocessor = ColorPostprocessor(stats_path, tolerance_percent=10.0, max_iterations=3)
        
        # Настройка логирования
        self.logger = setup_logger('DiffusionGenerator')
        
        # Загружаем доступные модели
        self.available_models = self._get_available_models()
        
        self.logger.info(f"Генератор инициализирован на устройстве: {self.device}")
        self.logger.info(f"Доступные модели: {list(self.available_models.keys())}")
    
    def _get_available_models(self) -> dict:
        """Получает список доступных моделей из чекпоинтов"""
        models = {}
        
        if not os.path.exists(self.checkpoint_dir):
            self.logger.warning(f"Директория чекпоинтов не найдена: {self.checkpoint_dir}")
            return models
        
        # Ищем файлы .pth
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pth'):
                # Извлекаем название класса из имени файла
                if file.startswith('unet_') and file.endswith('_best.pth'):
                    class_name = file[5:-9]  # Убираем 'unet_' и '_best.pth'
                    models[class_name] = os.path.join(self.checkpoint_dir, file)
        
        return models
    
    def load_model(self, class_name: str) -> UNet2DModel:
        """Загружает модель для конкретного класса"""
        if class_name not in self.available_models:
            raise ValueError(f"Модель для класса {class_name} не найдена. "
                           f"Доступные: {list(self.available_models.keys())}")
        
        checkpoint_path = self.available_models[class_name]
        self.logger.info(f"Загружаю модель для класса {class_name} из {checkpoint_path}")
        
        # Создаем модель
        model = UNet2DModel(
            sample_size=128,
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
        
        # Загружаем веса
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Модель для класса {class_name} загружена успешно")
        return model
    
    def generate_single_image(self, class_name: str, output_path: str, 
                             postprocess: bool = True) -> str:
        """
        Генерирует одно изображение для указанного класса
        
        Args:
            class_name: Название класса для генерации
            output_path: Путь для сохранения изображения
            postprocess: Применять ли постобработку
            
        Returns:
            Путь к сохраненному изображению
        """
        try:
            # Загружаем модель
            model = self.load_model(class_name)
            
            # Создаем планировщик
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
            
            # Генерируем изображение
            self.logger.info(f"Генерирую изображение для класса {class_name}")
            
            # Создаем случайный шум
            noise = torch.randn(1, 3, 128, 128).to(self.device)
            
            # Процесс генерации
            latents = noise
            for t in scheduler.timesteps:
                # Предсказание шума
                with torch.no_grad():
                    noise_pred = model(latents, timestep=t).sample
                
                # Обновление латентов
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Конвертируем в изображение
            image = latents.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Применяем постобработку если нужно
            if postprocess:
                self.logger.info("Применяю постобработку цветов")
                image = self.postprocessor.postprocess_image(image, class_name)
            
            # Сохраняем изображение
            pil_image = Image.fromarray(image)
            pil_image.save(output_path)
            
            self.logger.info(f"Изображение сохранено: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации изображения для класса {class_name}: {e}")
            raise
    
    def generate_batch_images(self, class_name: str, output_dir: str, count: int,
                             postprocess: bool = True, batch_size: int = 4) -> List[str]:
        """
        Генерирует пакет изображений для указанного класса
        
        Args:
            class_name: Название класса для генерации
            output_dir: Директория для сохранения изображений
            count: Количество изображений для генерации
            postprocess: Применять ли постобработку
            batch_size: Размер пакета для генерации
            
        Returns:
            Список путей к сохраненным изображениям
        """
        try:
            # Создаем директорию если не существует
            os.makedirs(output_dir, exist_ok=True)
            
            # Загружаем модель
            model = self.load_model(class_name)
            
            # Создаем планировщик
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
            
            self.logger.info(f"Генерирую {count} изображений для класса {class_name}")
            
            generated_paths = []
            
            # Генерируем изображения пакетами
            for batch_start in range(0, count, batch_size):
                batch_end = min(batch_start + batch_size, count)
                batch_count = batch_end - batch_start
                
                self.logger.info(f"Генерирую пакет {batch_start//batch_size + 1}: "
                               f"изображения {batch_start + 1}-{batch_end}")
                
                # Создаем случайный шум для пакета
                noise = torch.randn(batch_count, 3, 128, 128).to(self.device)
                
                # Процесс генерации
                latents = noise
                for t in scheduler.timesteps:
                    # Предсказание шума
                    with torch.no_grad():
                        noise_pred = model(latents, timestep=t).sample
                    
                    # Обновление латентов
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                
                # Конвертируем в изображения
                images = latents.permute(0, 2, 3, 1).cpu().numpy()
                images = ((images + 1) * 127.5).clip(0, 255).astype(np.uint8)
                
                # Сохраняем каждое изображение
                for i, image in enumerate(images):
                    image_index = batch_start + i + 1
                    
                    # Применяем постобработку если нужно
                    if postprocess:
                        image = self.postprocessor.postprocess_image(image, class_name)
                    
                    # Сохраняем изображение
                    output_path = os.path.join(output_dir, f"{class_name}_{image_index:04d}.jpg")
                    pil_image = Image.fromarray(image)
                    pil_image.save(output_path)
                    
                    generated_paths.append(output_path)
                
                # Очищаем память CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.logger.info(f"Сгенерировано {len(generated_paths)} изображений")
            return generated_paths
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации пакета изображений для класса {class_name}: {e}")
            raise
    
    def get_available_classes(self) -> List[str]:
        """Возвращает список доступных классов"""
        return list(self.available_models.keys())
    
    def get_model_info(self, class_name: str) -> Optional[Dict]:
        """Возвращает информацию о модели для класса"""
        if class_name in self.available_models:
            return {
                'class_name': class_name,
                'checkpoint_path': self.available_models[class_name],
                'device': str(self.device)
            }
        return None
