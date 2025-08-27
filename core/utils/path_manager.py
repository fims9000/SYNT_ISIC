"""
PathManager - управление путями и файлами для ISIC Generator
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import logging

class PathManager:
    """Менеджер путей и файлов для ISIC Generator"""
    
    def __init__(self, base_dir: str = None):
        """
        Инициализация менеджера путей
        
        Args:
            base_dir: Базовая директория (по умолчанию - текущая рабочая)
        """
        # Определяем корень проекта относительно расположения файла (…/core/utils/ -> проект)
        default_project_root = Path(__file__).resolve().parents[2]
        self.base_dir = Path(base_dir).resolve() if base_dir else default_project_root
        self.logger = logging.getLogger(__name__)
        
    def get_absolute_path(self, relative_path: str) -> Path:
        """Получает абсолютный путь относительно базовой директории"""
        return (self.base_dir / relative_path).resolve()
    
    def ensure_dir(self, path: str) -> Path:
        """Создает директорию если она не существует"""
        full_path = self.get_absolute_path(path)
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path
    
    def get_checkpoint_path(self, class_name: str) -> Optional[Path]:
        """Получает путь к чекпоинту для указанного класса"""
        checkpoint_dir = self.get_absolute_path("checkpoints")
        checkpoint_file = checkpoint_dir / f"unet_{class_name}_best.pth"
        
        if checkpoint_file.exists():
            return checkpoint_file
        return None
    
    def get_available_classes(self) -> List[str]:
        """Получает список доступных классов на основе чекпоинтов"""
        checkpoint_dir = self.get_absolute_path("checkpoints")
        if not checkpoint_dir.exists():
            return []
        
        classes = []
        for file in checkpoint_dir.glob("unet_*_best.pth"):
            # Извлекаем имя класса из имени файла надёжно
            class_name = file.stem.replace("unet_", "").replace("_best", "")
            if class_name:
                classes.append(class_name)
        
        return sorted(classes)
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """Получает директорию для вывода"""
        output_base = self.get_absolute_path("generated_images")
        if subdir:
            output_dir = output_base / subdir
        else:
            output_dir = output_base
        
        return self.ensure_dir(str(output_dir))
    
    def get_cache_dir(self) -> Path:
        """Получает директорию кэша"""
        return self.ensure_dir("core/cache")
    
    def get_logs_dir(self) -> Path:
        """Получает директорию логов"""
        return self.ensure_dir("core/logs")
    
    def get_temp_dir(self) -> Path:
        """Получает временную директорию"""
        temp_dir = self.get_cache_dir() / "temp"
        return self.ensure_dir(str(temp_dir))
    
    def create_unique_filename(self, base_name: str, extension: str = ".png") -> str:
        """Создает уникальное имя файла"""
        counter = 1
        filename = f"{base_name}{extension}"
        
        while self.get_absolute_path(filename).exists():
            filename = f"{base_name}_{counter}{extension}"
            counter += 1
        
        return filename
    
    def get_isic_filename(self, isic_number: int) -> str:
        """Создает имя файла в формате ISIC (png)"""
        return f"ISIC_{isic_number:07d}.png"
    
    def get_next_isic_number(self, output_dir: str = "generated_images") -> int:
        """Получает следующий номер ISIC для синтетического датасета в указанной папке.

        Если передан абсолютный путь, используется он. Иначе — путь относительно
        базовой директории через get_output_dir.
        """
        output_path = Path(output_dir) if os.path.isabs(output_dir) else self.get_output_dir(output_dir)
        max_number = 0
        
        # Ищем существующие файлы ISIC (поддерживаем .png и .jpg на всякий случай)
        candidates = list(output_path.glob("ISIC_*.png")) + list(output_path.glob("ISIC_*.jpg"))
        for file in candidates:
            try:
                number_str = file.stem.split("_")[1]
                number = int(number_str)
                max_number = max(max_number, number)
            except (ValueError, IndexError):
                continue
        
        return max_number + 1
    
    def cleanup_temp_files(self):
        """Очищает временные файлы"""
        temp_dir = self.get_temp_dir()
        try:
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            self.logger.info("Временные файлы очищены")
        except Exception as e:
            self.logger.error(f"Ошибка очистки временных файлов: {e}")
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Получает размер файла в мегабайтах"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0
    
    def get_directory_size_mb(self, dir_path: str) -> float:
        """Получает размер директории в мегабайтах"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)
        except OSError:
            return 0.0
    
    def copy_file(self, src: str, dst: str) -> bool:
        """Копирует файл"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка копирования файла {src} -> {dst}: {e}")
            return False
    
    def move_file(self, src: str, dst: str) -> bool:
        """Перемещает файл"""
        try:
            shutil.move(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка перемещения файла {src} -> {dst}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Удаляет файл"""
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка удаления файла {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[dict]:
        """Получает информацию о файле"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            return {
                'name': path.name,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_file': path.is_file(),
                'is_dir': path.is_dir()
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о файле {file_path}: {e}")
            return None
    
    def find_files_by_pattern(self, pattern: str, search_dir: str = "") -> List[Path]:
        """Ищет файлы по паттерну"""
        search_path = self.get_absolute_path(search_dir) if search_dir else self.base_dir
        return list(search_path.glob(pattern))
    
    def get_relative_path(self, absolute_path: str) -> str:
        """Получает относительный путь от базовой директории"""
        try:
            return str(Path(absolute_path).relative_to(self.base_dir))
        except ValueError:
            return absolute_path

