"""
CacheManager - управление кэшем для ISIC Generator
"""

import os
import shutil
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

class CacheManager:
    """Менеджер кэша для ISIC Generator"""
    
    def __init__(self, cache_dir: str = "core/cache"):
        """
        Инициализация менеджера кэша
        
        Args:
            cache_dir: Директория для кэша
        """
        # Разрешаем путь к кэшу относительно корня проекта, если путь относительный
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if os.path.isabs(cache_dir) else (project_root / cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Поддиректории кэша
        self.models_cache = self.cache_dir / "models"
        self.temp_cache = self.cache_dir / "temp"
        self.metadata_cache = self.cache_dir / "metadata"
        
        # Создаем поддиректории
        for subdir in [self.models_cache, self.temp_cache, self.metadata_cache]:
            subdir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.metadata_file = self.metadata_cache / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Загружает метаданные кэша"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки метаданных кэша: {e}")
        
        return {
            "models": {},
            "temp_files": {},
            "last_cleanup": datetime.now().isoformat(),
            "cache_stats": {
                "total_size_mb": 0,
                "models_count": 0,
                "temp_files_count": 0
            }
        }
    
    def _save_metadata(self):
        """Сохраняет метаданные кэша"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения метаданных кэша: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Вычисляет хеш файла"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Ошибка вычисления хеша файла {file_path}: {e}")
            return ""
    
    def cache_model(self, model_path: str, class_name: str) -> str:
        """
        Кэширует модель
        
        Args:
            model_path: Путь к модели
            class_name: Имя класса
            
        Returns:
            Путь к кэшированной модели
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
            
            # Вычисляем хеш модели
            model_hash = self._calculate_file_hash(model_path)
            if not model_hash:
                raise ValueError("Не удалось вычислить хеш модели")
            
            # Создаем имя кэшированного файла
            cached_name = f"{class_name}_{model_hash[:8]}.pth"
            cached_path = self.models_cache / cached_name
            
            # Проверяем, есть ли уже в кэше
            if cached_path.exists():
                self.logger.info(f"Модель {class_name} уже в кэше: {cached_path}")
                return str(cached_path)
            
            # Копируем модель в кэш
            shutil.copy2(model_path, cached_path)
            
            # Обновляем метаданные
            file_size = os.path.getsize(cached_path)
            self.metadata["models"][class_name] = {
                "cached_path": str(cached_path),
                "original_path": model_path,
                "hash": model_hash,
                "size_bytes": file_size,
                "cached_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat()
            }
            
            self._update_cache_stats()
            self._save_metadata()
            
            self.logger.info(f"Модель {class_name} закэширована: {cached_path}")
            return str(cached_path)
            
        except Exception as e:
            self.logger.error(f"Ошибка кэширования модели {class_name}: {e}")
            raise
    
    def get_cached_model(self, class_name: str) -> Optional[str]:
        """
        Получает кэшированную модель
        
        Args:
            class_name: Имя класса
            
        Returns:
            Путь к кэшированной модели или None
        """
        if class_name not in self.metadata["models"]:
            return None
        
        model_info = self.metadata["models"][class_name]
        cached_path = model_info["cached_path"]
        
        # Проверяем существование файла
        if not os.path.exists(cached_path):
            # Удаляем из метаданных если файл не найден
            del self.metadata["models"][class_name]
            self._save_metadata()
            return None
        
        # Обновляем время последнего доступа
        model_info["last_accessed"] = datetime.now().isoformat()
        self._save_metadata()
        
        return cached_path
    
    def is_model_cached(self, class_name: str) -> bool:
        """Проверяет, закэширована ли модель"""
        return class_name in self.metadata["models"]
    
    def create_temp_file(self, prefix: str = "", suffix: str = ".tmp") -> str:
        """
        Создает временный файл
        
        Args:
            prefix: Префикс имени файла
            suffix: Суффикс имени файла
            
        Returns:
            Путь к временному файлу
        """
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=suffix,
            dir=self.temp_cache,
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Добавляем в метаданные
        file_size = os.path.getsize(temp_path)
        temp_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata["temp_files"][temp_id] = {
            "path": temp_path,
            "size_bytes": file_size,
            "created_at": datetime.now().isoformat(),
            "prefix": prefix,
            "suffix": suffix
        }
        
        self._update_cache_stats()
        self._save_metadata()
        
        return temp_path
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Очищает старые временные файлы
        
        Args:
            max_age_hours: Максимальный возраст файлов в часах
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        files_to_remove = []
        
        for temp_id, temp_info in self.metadata["temp_files"].items():
            created_at = datetime.fromisoformat(temp_info["created_at"])
            if created_at < cutoff_time:
                files_to_remove.append(temp_id)
        
        for temp_id in files_to_remove:
            temp_info = self.metadata["temp_files"][temp_id]
            temp_path = temp_info["path"]
            
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                del self.metadata["temp_files"][temp_id]
                self.logger.info(f"Удален временный файл: {temp_path}")
            except Exception as e:
                self.logger.error(f"Ошибка удаления временного файла {temp_path}: {e}")
        
        if files_to_remove:
            self._update_cache_stats()
            self._save_metadata()
    
    def cleanup_old_models(self, max_age_days: int = 30):
        """
        Очищает старые кэшированные модели
        
        Args:
            max_age_days: Максимальный возраст моделей в днях
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        models_to_remove = []
        
        for class_name, model_info in self.metadata["models"].items():
            cached_at = datetime.fromisoformat(model_info["cached_at"])
            if cached_at < cutoff_time:
                models_to_remove.append(class_name)
        
        for class_name in models_to_remove:
            model_info = self.metadata["models"][class_name]
            cached_path = model_info["cached_path"]
            
            try:
                if os.path.exists(cached_path):
                    os.remove(cached_path)
                del self.metadata["models"][class_name]
                self.logger.info(f"Удалена старая кэшированная модель: {class_name}")
            except Exception as e:
                self.logger.error(f"Ошибка удаления кэшированной модели {class_name}: {e}")
        
        if models_to_remove:
            self._update_cache_stats()
            self._save_metadata()
    
    def _update_cache_stats(self):
        """Обновляет статистику кэша"""
        total_size = 0
        models_count = len(self.metadata["models"])
        temp_files_count = len(self.metadata["temp_files"])
        
        # Подсчитываем общий размер
        for model_info in self.metadata["models"].values():
            total_size += model_info.get("size_bytes", 0)
        
        for temp_info in self.metadata["temp_files"].values():
            total_size += temp_info.get("size_bytes", 0)
        
        self.metadata["cache_stats"] = {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "models_count": models_count,
            "temp_files_count": temp_files_count
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получает статистику кэша"""
        self._update_cache_stats()
        return self.metadata["cache_stats"].copy()
    
    def clear_all_cache(self):
        """Очищает весь кэш"""
        try:
            # Удаляем все файлы
            for subdir in [self.models_cache, self.temp_cache]:
                if subdir.exists():
                    shutil.rmtree(subdir)
                    subdir.mkdir()
            
            # Очищаем метаданные
            self.metadata = {
                "models": {},
                "temp_files": {},
                "last_cleanup": datetime.now().isoformat(),
                "cache_stats": {
                    "total_size_mb": 0,
                    "models_count": 0,
                    "temp_files_count": 0
                }
            }
            
            self._save_metadata()
            self.logger.info("Весь кэш очищен")
            
        except Exception as e:
            self.logger.error(f"Ошибка очистки кэша: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получает подробную информацию о кэше"""
        cache_info = {
            "cache_directory": str(self.cache_dir),
            "models_cache": str(self.models_cache),
            "temp_cache": str(self.temp_cache),
            "metadata_cache": str(self.metadata_cache),
            "stats": self.get_cache_stats(),
            "models": {},
            "temp_files": {}
        }
        
        # Информация о моделях
        for class_name, model_info in self.metadata["models"].items():
            cache_info["models"][class_name] = {
                "size_mb": round(model_info.get("size_bytes", 0) / (1024 * 1024), 2),
                "cached_at": model_info.get("cached_at", ""),
                "last_accessed": model_info.get("last_accessed", ""),
                "hash": model_info.get("hash", "")[:16] + "..."  # Первые 16 символов
            }
        
        # Информация о временных файлах
        for temp_id, temp_info in self.metadata["temp_files"].items():
            cache_info["temp_files"][temp_id] = {
                "size_mb": round(temp_info.get("size_bytes", 0) / (1024 * 1024), 2),
                "created_at": temp_info.get("created_at", ""),
                "prefix": temp_info.get("prefix", ""),
                "suffix": temp_info.get("suffix", "")
            }
        
        return cache_info
    
    def cleanup_cache(self, max_age_days: int = 30, max_age_hours: int = 24):
        """
        Выполняет полную очистку кэша
        
        Args:
            max_age_days: Максимальный возраст моделей в днях
            max_age_hours: Максимальный возраст временных файлов в часах
        """
        self.logger.info("Начинаю очистку кэша...")
        
        # Очищаем старые модели
        self.cleanup_old_models(max_age_days)
        
        # Очищаем старые временные файлы
        self.cleanup_temp_files(max_age_hours)
        
        # Обновляем время последней очистки
        self.metadata["last_cleanup"] = datetime.now().isoformat()
        self._save_metadata()
        
        self.logger.info("Очистка кэша завершена")
    
    def close(self):
        """Закрывает менеджер кэша"""
        try:
            # Выполняем финальную очистку
            self.cleanup_temp_files(max_age_hours=1)
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии менеджера кэша: {e}")

