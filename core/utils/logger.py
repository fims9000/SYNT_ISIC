# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Callable

DEFAULT_LOGGER_NAME = "ISICGenerator"

class Logger:
    """
    Универсальный логгер проекта.
    Совместим со старыми вызовами: log_info, log_warning, log_error, log_debug.
    Пишет в консоль и (опционально) в файл. Можно подвесить GUI‑коллбек.
    """

    def __init__(
        self,
        name: str = DEFAULT_LOGGER_NAME,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        max_bytes: int = 2 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        self.name = name
        self._callback: Optional[Callable[[str], None]] = None

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # чтобы не дублировать в root

        # Форматтер единый для всех хендлеров
        fmt = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%%Y-%%m-%%d %%H:%%M:%%S",
        )

        # Консоль
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(level)
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)

        # Файл
        try:
            if log_file is None:
                # если не передали, но указан каталог — лог туда; иначе рядом с исполняемым
                base_dir = Path(log_dir) if log_dir else Path.cwd()
                base_dir.mkdir(parents=True, exist_ok=True)
                log_file = str(base_dir / "app.log")
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            # избегаем дублирования файлового хендлера при повторной инициализации
            if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                self.logger.addHandler(fh)
        except Exception:
            # не падаем, если нет прав на запись
            pass

    # --- Совместимые методы ---
    def log_info(self, msg: str) -> None:
        self._emit(logging.INFO, msg)

    def log_warning(self, msg: str) -> None:
        self._emit(logging.WARNING, msg)

    def log_error(self, msg: str) -> None:
        self._emit(logging.ERROR, msg)

    def log_debug(self, msg: str) -> None:
        self._emit(logging.DEBUG, msg)

    # --- Дополнительно ---
    def set_gui_callback(self, cb: Callable[[str], None]) -> None:
        """Регистрирует коллбек GUI; он будет получать готовую строку лога."""
        self._callback = cb

    def setup_gui_handler(self, text_widget):
        """Привязывает вывод логов к QTextEdit в GUI."""
        try:
            def gui_callback(line: str):
                try:
                    text_widget.append(line)
                except Exception:
                    pass
            self.set_gui_callback(gui_callback)
        except Exception:
            pass

    def get_logger(self) -> logging.Logger:
        """Доступ к нативному logging.Logger при необходимости."""
        return self.logger

    # --- Внутреннее ---
    def _emit(self, level: int, msg: str) -> None:
        try:
            if level == logging.INFO:
                self.logger.info(msg)
            elif level == logging.WARNING:
                self.logger.warning(msg)
            elif level == logging.ERROR:
                self.logger.error(msg)
            elif level == logging.DEBUG:
                self.logger.debug(msg)
            else:
                self.logger.log(level, msg)
        finally:
            if self._callback is not None:
                try:
                    ts = datetime.now().strftime("%H:%M:%S")
                    level_name = logging.getLevelName(level)
                    line = f"{ts} - {self.name} - {level_name} - {msg}"
                    self._callback(line)
                except Exception:
                    # GUI не должен ломать логирование
                    pass

__all__ = ["Logger"]
