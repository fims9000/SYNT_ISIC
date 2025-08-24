"""
Logger - система логирования для ISIC Generator
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class Logger:
    """Система логирования для ISIC Generator"""
    
    def __init__(self, name: str = "ISICGenerator", log_dir: str = "core/logs"):
        """
        Инициализация логгера
        
        Args:
            name: Имя логгера
            log_dir: Директория для логов
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем основной логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Очищаем существующие хендлеры
        self.logger.handlers.clear()
        
        # Настраиваем форматирование
        self._setup_formatters()
        
        # Настраиваем хендлеры
        self._setup_handlers()
        
        # Словарь для хранения дополнительных логгеров
        self.special_loggers = {}
        
    def _setup_formatters(self):
        """Настраивает форматирование логов"""
        # Подробный формат для файлов
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Простой формат для консоли
        self.simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Формат для GUI
        self.gui_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _setup_handlers(self):
        """Настраивает хендлеры для логгера"""
        # Хендлер для основного файла логов
        main_log_file = self.log_dir / "generator.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(main_handler)
        
        # Хендлер для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Хендлер для ошибок
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # Хендлер для GUI (будет настроен позже)
        self.gui_handler = None
    
    def setup_gui_handler(self, text_widget):
        """
        Настраивает хендлер для отображения логов в GUI
        
        Args:
            text_widget: Виджет для отображения логов (QTextEdit)
        """
        from PyQt5.QtCore import QObject, pyqtSignal
        
        class QTextEditHandler(logging.Handler, QObject):
            log_message = pyqtSignal(str)
            
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                QObject.__init__(self)
                self.text_widget = text_widget
                self.log_message.connect(self._append_log)
            
            def emit(self, record):
                msg = self.format(record)
                self.log_message.emit(msg)
            
            def _append_log(self, message):
                if hasattr(self.text_widget, 'append'):
                    self.text_widget.append(message)
        
        # Удаляем старый GUI хендлер если есть
        if self.gui_handler:
            self.logger.removeHandler(self.gui_handler)
        
        # Создаем новый GUI хендлер
        self.gui_handler = QTextEditHandler(text_widget)
        self.gui_handler.setLevel(logging.INFO)
        self.gui_handler.setFormatter(self.gui_formatter)
        self.logger.addHandler(self.gui_handler)
    
    def get_special_logger(self, name: str, log_file: str = None) -> logging.Logger:
        """
        Получает специальный логгер для конкретной задачи
        
        Args:
            name: Имя логгера
            log_file: Имя файла для логов (опционально)
        
        Returns:
            Специальный логгер
        """
        if name in self.special_loggers:
            return self.special_loggers[name]
        
        # Создаем новый логгер
        special_logger = logging.getLogger(f"{self.name}.{name}")
        special_logger.setLevel(logging.DEBUG)
        
        # Настраиваем файл для специального логгера
        if log_file:
            log_path = self.log_dir / log_file
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=5*1024*1024,  # 5 MB
                backupCount=3,
                encoding='utf-8'
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(self.detailed_formatter)
            special_logger.addHandler(handler)
        
        self.special_loggers[name] = special_logger
        return special_logger
    
    def log_generation_start(self, class_name: str, count: int, output_dir: str):
        """Логирует начало генерации"""
        self.logger.info(f"Генерация начата: {class_name} x{count} -> {output_dir}")
    
    def log_generation_progress(self, class_name: str, current: int, total: int):
        """Логирует прогресс генерации"""
        percentage = (current / total) * 100
        self.logger.info(f"📊 Прогресс {class_name}: {current}/{total} ({percentage:.1f}%)")
    
    def log_generation_complete(self, class_name: str, count: int, output_dir: str):
        """Логирует завершение генерации"""
        self.logger.info(f"Генерация завершена: {class_name} x{count} -> {output_dir}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Логирует ошибку"""
        error_msg = f"ОШИБКА {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_warning(self, message: str):
        """Логирует предупреждение"""
        self.logger.warning(f"⚠️  {message}")
    
    def log_info(self, message: str):
        """Логирует информационное сообщение"""
        self.logger.info(f"ИНФО: {message}")
    
    def log_debug(self, message: str):
        """Логирует отладочное сообщение"""
        self.logger.debug(f"🔍 {message}")
    
    def log_success(self, message: str):
        """Логирует успешное выполнение"""
        self.logger.info(f"УСПЕХ: {message}")
    
    def log_config_change(self, setting: str, old_value: Any, new_value: Any):
        """Логирует изменение конфигурации"""
        self.logger.info(f"⚙️  Конфигурация изменена: {setting} = {old_value} -> {new_value}")
    
    def log_model_loaded(self, class_name: str, model_path: str):
        """Логирует загрузку модели"""
        self.logger.info(f"Модель загружена: {class_name} <- {model_path}")
    
    def log_model_error(self, class_name: str, error: str):
        """Логирует ошибку загрузки модели"""
        self.logger.error(f"Ошибка загрузки модели {class_name}: {error}")
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Очищает старые логи"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
                    self.logger.info(f"🗑️  Удален старый лог: {log_file.name}")
        except Exception as e:
            self.logger.error(f"Ошибка очистки старых логов: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Получает сводку по логам"""
        try:
            main_log_file = self.log_dir / "generator.log"
            if not main_log_file.exists():
                return {"error": "Файл логов не найден"}
            
            # Подсчитываем количество строк по уровням
            level_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}
            
            with open(main_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    for level in level_counts:
                        if f" - {level} - " in line:
                            level_counts[level] += 1
                            break
            
            # Получаем размер файла
            file_size_mb = main_log_file.stat().st_size / (1024 * 1024)
            
            return {
                "total_lines": sum(level_counts.values()),
                "level_counts": level_counts,
                "file_size_mb": round(file_size_mb, 2),
                "last_modified": datetime.fromtimestamp(main_log_file.stat().st_mtime).isoformat()
            }
        except Exception as e:
            return {"error": f"Ошибка чтения сводки логов: {e}"}
    
    def set_level(self, level: str):
        """Устанавливает уровень логирования"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
            self.log_info(f"Уровень логирования изменен на: {level.upper()}")
        else:
            self.log_warning(f"Неизвестный уровень логирования: {level}")
    
    def close(self):
        """Закрывает все хендлеры"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        for special_logger in self.special_loggers.values():
            for handler in special_logger.handlers[:]:
                handler.close()
                special_logger.removeHandler(handler)
