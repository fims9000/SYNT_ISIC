"""
ConfigManager - управление конфигурацией и путями для ISIC Generator
Адаптирован для работы на любой машине
"""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Менеджер конфигурации для ISIC Generator"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Инициализация менеджера конфигурации
        
        Args:
            config_file: Путь к файлу конфигурации (опционально)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
        self._setup_paths()
        self._setup_logging()
        
    def _get_default_config_path(self) -> str:
        """Получает путь к файлу конфигурации по умолчанию"""
        # Определяем ОС и создаем соответствующие пути
        if platform.system() == "Windows":
            config_dir = os.path.join(os.getenv('APPDATA', ''), 'ISICGenerator')
        elif platform.system() == "Darwin":  # macOS
            config_dir = os.path.expanduser('~/Library/Application Support/ISICGenerator')
        else:  # Linux и другие Unix-системы
            config_dir = os.path.expanduser('~/.config/ISICGenerator')
        
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, 'config.json')
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла или создает по умолчанию"""
        default_config = {
            "paths": {
                "checkpoints": "checkpoints",
                "output": "generated_images",
                "cache": "core/cache",
                "logs": "core/logs",
                "models": "models"
            },
            "generation": {
                "image_size": 128,
                "train_timesteps": 1000,
                "inference_timesteps": 50,
                "batch_size": 1,
                "seed_mode": "random",  # "random" или "fixed"
                "seed_value": 42,
                "xai_frequency": 1
            },
            "ui": {
                "theme": "light",
                "language": "ru",
                "auto_save": True
            },
            "advanced": {
                "enable_color_postprocessing": True,
                "enable_xai": False,
                "max_concurrent_generations": 2
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Объединяем с дефолтными настройками
                    self._merge_configs(default_config, user_config)
            except Exception as e:
                logging.warning(f"Ошибка загрузки конфигурации: {e}, используем настройки по умолчанию")
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict):
        """Рекурсивно объединяет пользовательскую и дефолтную конфигурации"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def _setup_paths(self):
        """Настраивает пути относительно корня проекта, а не CWD"""
        # Корень проекта: два уровня вверх от этого файла: core/config/ -> проект
        base_dir = str(Path(__file__).resolve().parents[2])
        
        # Обновляем пути, делая их относительными к базовой директории
        for path_key in self.config["paths"]:
            val = self.config["paths"][path_key]
            if not os.path.isabs(val):
                self.config["paths"][path_key] = os.path.join(base_dir, val)
        
        # Создаем необходимые директории
        for path in self.config["paths"].values():
            os.makedirs(path, exist_ok=True)
    
    def _setup_logging(self):
        """Настраивает систему логирования"""
        log_dir = self.config["paths"]["logs"]
        log_file = os.path.join(log_dir, "generator.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def get_path(self, path_key: str) -> str:
        """Получает путь по ключу"""
        if path_key not in self.config["paths"]:
            raise KeyError(f"Неизвестный ключ пути: {path_key}")
        return self.config["paths"][path_key]
    
    def get_generation_param(self, param_key: str, default: Any = None) -> Any:
        """Получает параметр генерации по ключу"""
        if param_key not in self.config["generation"]:
            if default is not None:
                return default
            raise KeyError(f"Неизвестный параметр генерации: {param_key}")
        return self.config["generation"][param_key]
    
    def get_ui_setting(self, setting_key: str) -> Any:
        """Получает настройку UI по ключу"""
        if setting_key not in self.config["ui"]:
            raise KeyError(f"Неизвестная настройка UI: {setting_key}")
        return self.config["ui"][setting_key]
    
    def get_advanced_setting(self, setting_key: str) -> Any:
        """Получает расширенную настройку по ключу"""
        if setting_key not in self.config["advanced"]:
            raise KeyError(f"Неизвестная расширенная настройка: {setting_key}")
        return self.config["advanced"][setting_key]
    
    def update_path(self, path_key: str, new_path: str):
        """Обновляет путь"""
        if path_key not in self.config["paths"]:
            raise KeyError(f"Неизвестный ключ пути: {path_key}")
        
        self.config["paths"][path_key] = new_path
        os.makedirs(new_path, exist_ok=True)
        self._save_config()
    
    def update_generation_param(self, param_key: str, new_value: Any):
        """Обновляет параметр генерации (создаёт новый, если не существует)"""
        self.config["generation"][param_key] = new_value
        self._save_config()
    
    def _save_config(self):
        """Сохраняет конфигурацию в файл"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")
    
    def get_all_paths(self) -> Dict[str, str]:
        """Возвращает все пути"""
        return self.config["paths"].copy()
    
    def get_all_generation_params(self) -> Dict[str, Any]:
        """Возвращает все параметры генерации"""
        return self.config["generation"].copy()
    
    def reset_to_defaults(self):
        """Сбрасывает конфигурацию к значениям по умолчанию"""
        self.config = self._load_config()
        self._setup_paths()
        self._save_config()
        logging.info("Конфигурация сброшена к значениям по умолчанию")
    
    def export_config(self, export_path: str):
        """Экспортирует конфигурацию в указанный файл"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logging.info(f"Конфигурация экспортирована в: {export_path}")
        except Exception as e:
            logging.error(f"Ошибка экспорта конфигурации: {e}")
    
    def import_config(self, import_path: str):
        """Импортирует конфигурацию из указанного файла"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            self._merge_configs(self.config, imported_config)
            self._setup_paths()
            self._save_config()
            logging.info(f"Конфигурация импортирована из: {import_path}")
        except Exception as e:
            logging.error(f"Ошибка импорта конфигурации: {e}")

