#!/usr/bin/env python3
"""
ISIC Synthetic Data Generator - GUI Interface
Интерфейс для генерации синтетических данных ISIC
"""

import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QCheckBox, QSpinBox, QProgressBar, QTextEdit, 
                             QTreeWidget, QTreeWidgetItem, QComboBox, QGroupBox, 
                             QFrame, QFileDialog, QMessageBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap

# Импортируем core пакет
try:
    from core import ConfigManager, ImageGenerator, Logger, PathManager, CacheManager
    import torch  # Добавляем импорт torch для работы с CUDA
except ImportError:
    print("Ошибка импорта core пакета. Убедитесь, что все файлы созданы.")
    sys.exit(1)

class GenerationWorker(QThread):
    """Воркер для асинхронной генерации изображений"""
    progress_updated = pyqtSignal(int, int, str)
    log_updated = pyqtSignal(str)
    generation_finished = pyqtSignal(dict)
    
    def __init__(self, generator, class_configs, output_dir):
        super().__init__()
        self.generator = generator
        self.class_configs = class_configs
        self.output_dir = output_dir
        
    def run(self):
        try:
            # Устанавливаем callback'и
            self.generator.set_progress_callback(self.progress_updated.emit)
            self.generator.set_log_callback(self.log_updated.emit)
            
            # Запускаем генерацию
            results = self.generator.generate_images(
                self.class_configs, 
                self.output_dir, 
                postprocess=True  # Постобработка всегда включена
            )
            
            self.generation_finished.emit(results)
            
        except Exception as e:
            self.log_updated.emit(f"ОШИБКА: Критическая ошибка: {str(e)}")
            self.generation_finished.emit({"error": str(e)})

class XAIWorker(QThread):
    """Воркер для запуска полного XAI пайплайна (скрипт xai/XAI.py)"""
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, working_dir: str):
        super().__init__()
        self.working_dir = working_dir
        self.proc = None
        self._stop_requested = False

    def run(self):
        try:
            self.log_updated.emit("XAI: запуск полного анализа (xai/XAI.py)...")
            python_exe = sys.executable or "python"
            script_path = os.path.join(self.working_dir, 'xai', 'XAI.py')
            if not os.path.exists(script_path):
                # альтернативный регистр папки
                script_path = os.path.join(self.working_dir, 'XAI', 'XAI.py')
            if not os.path.exists(script_path):
                self.log_updated.emit("XAI: скрипт не найден: xai/XAI.py")
                self.finished.emit(False)
                return

            # Гарантируем UTF-8 вывод, чтобы эмодзи/русский текст не падали в cp1251
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["MPLBACKEND"] = "Agg"

            proc = subprocess.Popen(
                [python_exe, '-u', script_path],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                env=env
            )
            self.proc = proc

            # Стримим логи
            assert proc.stdout is not None
            for line in proc.stdout:
                if line is not None:
                    self.log_updated.emit(line.rstrip())

            code = proc.wait()
            success = (code == 0)
            self.log_updated.emit(f"XAI: завершен с кодом {code}")
            self.finished.emit(success)
        except Exception as e:
            self.log_updated.emit(f"XAI: ошибка: {str(e)}")
            self.finished.emit(False)
        finally:
            self.proc = None

    def stop(self):
        try:
            self._stop_requested = True
            if self.proc and self.proc.poll() is None:
                self.log_updated.emit("XAI: terminating process...")
                self.proc.kill()
        except Exception:
            pass

class SyntheticDataGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Очищаем логи при запуске
        self._cleanup_logs_on_startup()
        
        # Инициализируем core компоненты
        try:
            self.config_manager = ConfigManager()
            self.path_manager = PathManager()
            self.logger = Logger()
            self.cache_manager = CacheManager()
            self.generator = ImageGenerator(self.config_manager)
            # Включаем XAI-хук: каждые 10 изображений
            self.generator.set_xai_hook(self._run_xai_for_image, every_n=10)
            # Выставляем общий seed (при желании можно сделать настраиваемым)
            self.generator.set_generation_seed(42)
        except Exception as e:
            QMessageBox.critical(None, "Ошибка инициализации", 
                               f"Не удалось инициализировать core компоненты: {str(e)}")
            sys.exit(1)
        
        # Состояние приложения
        self.is_generating = False
        self.generation_worker = None
        self.xai_worker = None
        self.selected_models_dir = ""
        self.selected_output_dir = ""
        
        # Состояние изображений
        self.current_image_path = None
        # Режим XAI (выключен по умолчанию)
        self.xai_mode = False
        
        # Загружаем доступные модели
        self.available_classes = self.generator.get_available_classes()
        
        self.init_ui()
        self.setup_connections()
        self.update_ui_state()
        
        # Настраиваем логирование в GUI
        self.logger.setup_gui_handler(self.logs_text)
        
        # Создаем таймер для обновления информации о памяти
        self.memory_update_timer = QTimer()
        self.memory_update_timer.timeout.connect(self.update_memory_info)
        self.memory_update_timer.start(2000)  # Обновляем каждые 2 секунды
        
        # Логируем запуск
        self.logs_text.append("Система инициализирована. Готова к генерации.")
        self.logs_text.append(f"Доступные модели: {len(self.available_classes)}")
        self.logs_text.append(f"Доступные классы: {', '.join(self.available_classes)}")
        
        # Обновляем информацию о памяти
        self.update_memory_info()
        
        # Показываем изображения если они есть
        if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
            self.show_first_generated_image()
        
    def update_memory_info(self):
        """Обновляет информацию о памяти"""
        try:
            # Обновляем только информацию о памяти
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                self.memory_info_label.setText(f"Память: {memory_allocated:.2f}ГБ / {memory_reserved:.2f}ГБ")
            else:
                self.memory_info_label.setText("Память: Режим CPU")
                
        except Exception as e:
            self.memory_info_label.setText("Память: Ошибка")
            self.logs_text.append(f"Ошибка обновления памяти: {str(e)}")
        
    def _cleanup_logs_on_startup(self):
        """Очищает логи при запуске программы"""
        try:
            import os
            from pathlib import Path
            
            # Пути к логам
            log_paths = [
                "core/logs/errors.log",
                "core/logs/generator.log", 
                "core/logs/test.log"
            ]
            
            for log_path in log_paths:
                if os.path.exists(log_path):
                    # Очищаем содержимое файла
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Log cleared on startup: {os.path.basename(log_path)}\n")
                        f.write(f"# Started at: {Path().absolute()}\n")
                        f.write("#" * 50 + "\n\n")
                        
        except Exception as e:
            # Игнорируем ошибки очистки логов
            print(f"Ошибка очистки логов: {e}")
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("ISIC Генератор Синтетических Данных")
        self.setGeometry(100, 100, 1200, 800)
        
        
        # Применяем стили
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #F8F8F8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #404040;
                font-size: 11pt;
            }
            QPushButton {
                background-color: #F0F0F0;
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                padding: 10px 20px;
                font-size: 10pt;
                min-height: 24px;
                color: #202020;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E8E8E8;
                border-color: #808080;
            }
            QPushButton:pressed {
                background-color: #D8D8D8;
                border-color: #606060;
            }
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #808080;
                border-color: #C0C0C0;
            }
            QCheckBox {
                spacing: 8px;
                font-size: 10pt;
                color: #333333;
                min-width: 120px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #C0C0C0;
                border-radius: 0px;
                background-color: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background-color: #000000;
                border-color: #000000;
            }
            QCheckBox::indicator:checked::after {
                content: "✕";
                color: #FFFFFF;
                font-weight: bold;
                font-size: 12px;
                text-align: center;
                line-height: 18px;
            }
            QSpinBox {
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                padding: 6px;
                min-height: 24px;
                background-color: #FFFFFF;
                color: #202020;
                font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                text-align: center;
                background-color: #F0F0F0;
                color: #202020;
                font-weight: bold;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background-color: #404040;
                border-radius: 0px;
            }
            QTextEdit {
                border: 2px solid #C0C0C0;
                border-radius: 0px;
                background-color: #FAFAFA;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                color: #333333;
                padding: 8px;
                line-height: 1.2;
            }
            QTreeWidget {
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                background-color: #FAFAFA;
                color: #202020;
                font-weight: bold;
            }
            QComboBox {
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                padding: 6px;
                min-height: 24px;
                background-color: #FFFFFF;
                color: #202020;
                font-weight: bold;
            }
            QLabel {
                margin: 2px;
                padding: 2px;
                color: #202020;
                font-weight: bold;
            }
        """)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем главный layout
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Создаем панели
        self.create_top_panel(main_layout)
        self.create_left_panel(main_layout)
        self.create_center_panel(main_layout)
        self.create_right_panel(main_layout)
        self.create_bottom_panel(main_layout)
        
        # Настраиваем пропорции строк - верхние панели получают больше места
        main_layout.setRowStretch(1, 3)  # Class Configuration - больше места
        main_layout.setRowStretch(2, 3)  # Generated Images - больше места  
        main_layout.setRowStretch(3, 1)  # Logs/Configuration - меньше места
        
        # Принудительно обновляем размеры
        self.updateGeometry()
        self.adjustSize()
        
    def create_top_panel(self, main_layout):
        """Создает верхнюю панель"""
        top_group = QGroupBox("Системные элементы управления")
        top_layout = QVBoxLayout(top_group)
        top_layout.setSpacing(25)
        
        # Строка с кнопками
        button_layout = QHBoxLayout()
        
        # Кнопка выбора модели
        self.select_model_btn = QPushButton("Выбрать модель")
        self.select_model_btn.setToolTip("Выберите папку с моделями (checkpoints)")
        
        # Кнопка выбора директории вывода
        self.select_output_btn = QPushButton("Выбрать папку вывода")
        self.select_output_btn.setToolTip("Выберите папку для сохранения изображений")
        
        # Тумблер XAI Mode
        self.xai_mode_btn = QPushButton("XAI Mode")
        self.xai_mode_btn.setCheckable(True)
        self.xai_mode_btn.setToolTip("Включить режим объяснимого ИИ")
        
        # ComboBox для выбора устройства с автоматическим определением
        self.device_combo = QComboBox()
        self._populate_device_combo()
        self.device_combo.setToolTip("Выберите устройство для генерации")
        
        # Контрол частоты XAI по шагам (n_steps)
        self.xai_step_spin = QSpinBox()
        self.xai_step_spin.setRange(1, 1000)
        self.xai_step_spin.setValue(50)  # по умолчанию как сейчас
        self.xai_step_spin.setToolTip("Сохранять XAI шаги каждые N timesteps (0..1000, включительно 1000)")
        
        # Добавляем кнопки в layout
        button_layout.addWidget(self.select_model_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.select_output_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.xai_mode_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(QLabel("XAI шаги:"))
        button_layout.addWidget(self.xai_step_spin)
        button_layout.addSpacing(10)
        button_layout.addWidget(QLabel("Устройство:"))
        button_layout.addSpacing(5)
        button_layout.addWidget(self.device_combo)
        
        top_layout.addLayout(button_layout)
        
        main_layout.addWidget(top_group, 0, 0, 1, 4)
        
    def _populate_device_combo(self):
        """Заполняет ComboBox доступными устройствами"""
        try:
            import torch
            
            # Добавляем CPU
            self.device_combo.addItem("CPU")
            
            # Проверяем доступность CUDA
            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    device_name = torch.cuda.get_device_name(i)
                    self.device_combo.addItem(f"CUDA:{i} ({device_name})")
                    
                # Устанавливаем первое CUDA устройство по умолчанию
                self.device_combo.setCurrentIndex(1)
            else:
                # Если CUDA недоступна, устанавливаем CPU
                self.device_combo.setCurrentIndex(0)
                
        except Exception as e:
            # В случае ошибки добавляем только CPU
            self.device_combo.addItem("CPU")
            self.device_combo.setCurrentIndex(0)
        
    def create_left_panel(self, main_layout):
        """Создает левую панель"""
        left_group = QGroupBox("Выбор и настройка классов")
        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(18)  # Увеличиваем междустрочный интервал
        
        # Создаем чекбоксы и спинбоксы для классов
        self.class_widgets = {}
        
        # Добавляем заголовок для классов
        class_header = QLabel("Доступные классы:")
        class_header.setStyleSheet("font-weight: bold; color: #404040; margin-bottom: 8px;")
        left_layout.addWidget(class_header)
        left_layout.addSpacing(5)
        
        for i, class_name in enumerate(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']):
            # Проверяем доступность модели
            is_available = class_name in self.available_classes
            
            class_layout = QHBoxLayout()
            class_layout.setSpacing(15)  # Увеличиваем расстояние между элементами
            
            # Чекбокс с фиксированной шириной
            checkbox = QCheckBox(class_name)
            checkbox.setEnabled(is_available)
            checkbox.setFixedWidth(140)  # Фиксируем ширину чекбокса
            if not is_available:
                checkbox.setToolTip(f"Модель для класса {class_name} недоступна")
            
            # Спинбокс
            spinbox = QSpinBox()
            spinbox.setRange(1, 10000)
            spinbox.setValue(5)
            spinbox.setEnabled(is_available)
            spinbox.setFixedWidth(80)  # Увеличиваем ширину спинбокса
            
            class_layout.addWidget(checkbox)
            class_layout.addStretch()  # Добавляем растягивающийся элемент
            class_layout.addWidget(QLabel("Количество:"))
            class_layout.addWidget(spinbox)
            
            left_layout.addLayout(class_layout)
            
            # Добавляем небольшой отступ между классами
            if i < len(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']) - 1:
                left_layout.addSpacing(3)
            
            self.class_widgets[class_name] = {
                'checkbox': checkbox,
                'spinbox': spinbox,
                'available': is_available
            }
        
        # Добавляем разделитель перед кнопками
        left_layout.addSpacing(10)
        
        # Заголовок для кнопок управления
        control_header = QLabel("Управление генерацией:")
        control_header.setStyleSheet("font-weight: bold; color: #404040; margin-bottom: 8px;")
        left_layout.addWidget(control_header)
        left_layout.addSpacing(5)
        
        # Кнопки управления
        self.start_btn = QPushButton("Начать генерацию")
        self.stop_btn = QPushButton("Остановить генерацию")
        self.regenerate_btn = QPushButton("Регенерировать")
        
        # Изначально Stop отключена
        self.stop_btn.setEnabled(False)
        
        left_layout.addWidget(self.start_btn)
        left_layout.addSpacing(5)
        left_layout.addWidget(self.stop_btn)
        left_layout.addSpacing(5)
        left_layout.addWidget(self.regenerate_btn)
        
        # Устанавливаем фиксированную ширину для левой панели
        left_group.setFixedWidth(280)
        
        main_layout.addWidget(left_group, 1, 0, 2, 1)
        
    def create_center_panel(self, main_layout):
        """Создает центральную панель"""
        center_group = QGroupBox("Предварительный просмотр генерации изображений")
        center_layout = QVBoxLayout(center_group)
        center_layout.setSpacing(18)
        
        # Placeholder для изображения
        self.image_label = QLabel("Предварительный просмотр сгенерированного изображения\n\nВыберите папку класса и файл изображения в правой панели")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #A0A0A0;
                border-radius: 0px;
                background-color: #FAFAFA;
                color: #404040;
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        
        # Делаем изображение кликабельным для открытия в полном размере
        self.image_label.mousePressEvent = self.on_image_clicked
        
        # Прогресс-бар
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Прогресс генерации:")
        progress_label.setStyleSheet("font-weight: bold; color: #404040;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        progress_layout.addWidget(progress_label)
        progress_layout.addSpacing(10)
        progress_layout.addWidget(self.progress_bar)
        
        center_layout.addWidget(self.image_label)
        center_layout.addLayout(progress_layout)
        
        main_layout.addWidget(center_group, 1, 1, 2, 2)
        
    def create_right_panel(self, main_layout):
        """Создает правую панель"""
        right_group = QGroupBox("Структура проекта")
        right_layout = QVBoxLayout(right_group)
        right_layout.setSpacing(18)
        
        # Дерево проекта
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabel("Компоненты проекта")
        
        # Заполняем дерево
        root_item = QTreeWidgetItem(self.project_tree, ["Проект синтетических данных"])
        self.generated_images_item = QTreeWidgetItem(root_item, ["generated_images"])
        self.xai_results_item = QTreeWidgetItem(root_item, ["xai_results"])
        self.checkpoints_item = QTreeWidgetItem(root_item, ["checkpoints"])
        
        # Подключаем обработчик кликов
        self.project_tree.itemClicked.connect(self.on_project_item_clicked)
        
        self.project_tree.expandAll()
        
        right_layout.addWidget(self.project_tree)
        
        # Список файлов изображений
        files_group = QGroupBox("Сгенерированные изображения")
        files_layout = QVBoxLayout(files_group)
        
        # Список папок классов
        self.class_folders_list = QListWidget()
        self.class_folders_list.setMaximumHeight(100)
        self.class_folders_list.itemClicked.connect(self.on_class_folder_clicked)
        
        # Список файлов изображений
        self.images_list = QListWidget()
        self.images_list.setMaximumHeight(150)
        self.images_list.itemClicked.connect(self.on_image_file_clicked)
        
        files_layout.addWidget(QLabel("Папки классов:"))
        files_layout.addWidget(self.class_folders_list)
        files_layout.addWidget(QLabel("Файлы изображений:"))
        files_layout.addWidget(self.images_list)
        
        right_layout.addWidget(files_group)

        # XAI Results panel
        xai_group = QGroupBox("Результаты XAI")
        xai_layout = QVBoxLayout(xai_group)
        
        self.xai_runs_list = QListWidget()
        self.xai_runs_list.setMaximumHeight(120)
        self.xai_runs_list.itemClicked.connect(self.on_xai_run_clicked)
        
        self.xai_files_list = QListWidget()
        self.xai_files_list.setMaximumHeight(180)
        self.xai_files_list.itemClicked.connect(self.on_xai_file_clicked)
        
        xai_layout.addWidget(QLabel("Запуски:"))
        xai_layout.addWidget(self.xai_runs_list)
        xai_layout.addWidget(QLabel("Файлы:"))
        xai_layout.addWidget(self.xai_files_list)
        
        right_layout.addWidget(xai_group)
        
        # Устанавливаем фиксированную ширину для правой панели
        right_group.setFixedWidth(250)
        
        main_layout.addWidget(right_group, 1, 3, 2, 1)
        
    def create_bottom_panel(self, main_layout):
        """Создает нижнюю панель"""
        # Панель логов
        logs_group = QGroupBox("Системные логи")
        logs_layout = QVBoxLayout(logs_group)
        logs_layout.setSpacing(18)
        
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMinimumHeight(150)  # Увеличиваем минимальную высоту
        self.logs_text.setMaximumHeight(250)  # Увеличиваем максимальную высоту
        
        # Теперь можем добавлять отладочные сообщения
        self.logs_text.append("Интерфейс успешно инициализирован")
        
        logs_layout.addWidget(self.logs_text)
        logs_layout.addSpacing(5)
        
        # Добавляем информацию о логах
        logs_info = QLabel("Системные логи и прогресс генерации будут отображаться здесь")
        logs_info.setStyleSheet("color: #606060; font-style: italic; font-size: 9pt;")
        logs_layout.addWidget(logs_info)
        
        # Панель конфигурации
        config_group = QGroupBox("Системная конфигурация")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(12)  # Увеличиваем междустрочный интервал
        
        # Статическая информация
        config_header = QLabel("Текущая конфигурация:")
        config_header.setStyleSheet("font-weight: bold; color: #404040; margin-bottom: 8px;")
        config_layout.addWidget(config_header)
        config_layout.addSpacing(5)
        
        self.device_info_label = QLabel("Устройство: CPU")
        self.model_path_label = QLabel("Путь к модели: Не выбран")
        self.available_models_label = QLabel(f"Доступные модели: {len(self.available_classes)}")
        self.color_config_label = QLabel("Цветовая конфигурация: Загружена")
        self.memory_info_label = QLabel("Память: Недоступна")
        
        config_layout.addWidget(self.device_info_label)
        config_layout.addSpacing(2)
        config_layout.addWidget(self.model_path_label)
        config_layout.addSpacing(2)
        config_layout.addWidget(self.available_models_label)
        config_layout.addSpacing(2)
        config_layout.addWidget(self.color_config_label)
        config_layout.addSpacing(2)
        config_layout.addWidget(self.memory_info_label)
        
        # Убираем фиксированную высоту, чтобы панель автоматически подстраивалась
        # config_group.setFixedHeight(80)
        
        main_layout.addWidget(logs_group, 3, 0, 1, 2)
        main_layout.addWidget(config_group, 3, 2, 1, 2)
        
    def setup_connections(self):
        """Настраивает соединения сигналов"""
        # Кнопки выбора папок
        self.select_model_btn.clicked.connect(self.select_models_directory)
        self.select_output_btn.clicked.connect(self.select_output_directory)
        
        # Кнопки управления генерацией
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.regenerate_btn.clicked.connect(self.on_regenerate_clicked)
        
        # ComboBox устройства
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        
        # Тумблер XAI Mode
        self.xai_mode_btn.toggled.connect(self.on_xai_toggle)
        
        # Инициализация XAI списков
        self.update_xai_lists()
        
    def on_project_item_clicked(self, item, column):
        """Обработчик кликов по элементам дерева проекта"""
        try:
            item_text = item.text(0)
            
            if item_text == "generated_images":
                self.open_generated_images_directory()
            elif item_text == "xai_results":
                self.update_xai_lists()
            elif item_text == "checkpoints":
                self.open_checkpoints_directory()
                
        except Exception as e:
            self.logs_text.append(f"Ошибка открытия папки: {str(e)}")
            
    def open_generated_images_directory(self):
        """Открывает папку с сгенерированными изображениями и показывает первое изображение"""
        try:
            if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                # Открываем папку в проводнике Windows
                os.startfile(self.selected_output_dir)
                self.logs_text.append(f"Открыта папка сгенерированных изображений: {self.selected_output_dir}")
                
                # Показываем первое найденное изображение в интерфейсе
                self.show_first_generated_image()
            else:
                QMessageBox.information(self, "Информация", "Сначала выберите папку для вывода!")
                
        except Exception as e:
            self.logs_text.append(f"Ошибка открытия папки сгенерированных изображений: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def show_first_generated_image(self):
        """Показывает первое найденное изображение в интерфейсе"""
        try:
            if not hasattr(self, 'selected_output_dir') or not self.selected_output_dir:
                return
                
            # Обновляем списки файлов
            self.update_file_lists()
            
            # Ищем первое изображение для показа
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            found_images = []
            
            for root, dirs, files in os.walk(self.selected_output_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        found_images.append(os.path.join(root, file))
                        
            if found_images:
                # Берем первое изображение
                first_image_path = sorted(found_images)[0]
                self.display_image(first_image_path)
                self.logs_text.append(f"Загружено {len(found_images)} изображений из папки вывода")
            else:
                self.image_label.setText("Предварительный просмотр сгенерированного изображения\n\nВ папке вывода не найдено изображений")
                
        except Exception as e:
            self.logs_text.append(f"Ошибка показа первого изображения: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def display_image(self, image_path):
        """Отображает изображение в интерфейсе в исходном размере"""
        try:
            from PIL import Image
            
            # Загружаем изображение
            pil_image = Image.open(image_path)
            
            # Если XAI режим включен — формируем оверлей поверх изображения
            if getattr(self, 'xai_mode', False):
                _xai_func = None
                try:
                    # 1) Пытаемся загрузить модуль напрямую по пути файла (надёжно при конфликтах пакетов)
                    import importlib.util
                    mod_path = os.path.join(os.getcwd(), 'xai', 'xai_integration.py')
                    if not os.path.exists(mod_path):
                        mod_path = os.path.join(os.getcwd(), 'XAI', 'xai_integration.py')
                    spec = importlib.util.spec_from_file_location('xai_xai_integration_dynamic', mod_path)
                    if spec and spec.loader:
                        dyn_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(dyn_mod)
                        _xai_func = getattr(dyn_mod, 'run_xai_analysis', None)
                        self.logs_text.append(f"XAI динамически загружен из: {mod_path}")
                except Exception as e0:
                    self.logs_text.append(f"Импорт XAI по пути не удался: {str(e0)}")
                
                if _xai_func is None:
                    # 2) Пробуем пакетный экспорт
                    try:
                        from xai import run_xai_analysis as _xai_func
                    except Exception as e1:
                        # 3) Пробуем импорт пакета и извлечение атрибута
                        try:
                            import importlib
                            mod = importlib.import_module('xai.xai_integration')
                            _xai_func = getattr(mod, 'run_xai_analysis')
                            try:
                                mod_path = getattr(mod, '__file__', 'unknown')
                                self.logs_text.append(f"XAI загружен из: {mod_path}")
                            except Exception:
                                pass
                        except Exception as e2:
                            _xai_func = None
                            self.logs_text.append(f"Резервный импорт XAI не удался: {str(e2)} (основной: {str(e1)})")
                if _xai_func is None:
                    # Третий вариант: прямой импорт по пути файла
                    try:
                        import importlib.util
                        mod_path = os.path.join(os.getcwd(), 'xai', 'xai_integration.py')
                        if not os.path.exists(mod_path):
                            mod_path = os.path.join(os.getcwd(), 'XAI', 'xai_integration.py')
                        spec = importlib.util.spec_from_file_location('xai_xai_integration_dynamic', mod_path)
                        if spec and spec.loader:
                            dyn_mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(dyn_mod)
                            _xai_func = getattr(dyn_mod, 'run_xai_analysis', None)
                            self.logs_text.append(f"XAI динамически загружен из: {mod_path}")
                    except Exception as e3:
                        self.logs_text.append(f"Импорт XAI по пути не удался: {str(e3)}")

                if _xai_func is not None:
                    try:
                        device = getattr(self.generator, 'device', None)
                        classifier_path = os.path.join(os.getcwd(), 'checkpoints', 'classifier.pth')
                        save_dir = os.path.join(os.getcwd(), 'xai_results')
                        overlay_pil, saved_path = _xai_func(
                            image_path,
                            device=device,
                            classifier_path=classifier_path,
                            save_dir=save_dir
                        )
                        pil_image = overlay_pil
                        self.logs_text.append(f"XAI оверлей сохранен: {saved_path}")
                    except Exception as e:
                        self.logs_text.append(f"Ошибка выполнения XAI (возврат к оригиналу): {str(e)}")
                else:
                    self.logs_text.append("Импорт XAI не разрешен; показывается оригинальное изображение")
            
            # Конвертируем в QPixmap
            from PyQt5.QtGui import QPixmap
            import io
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            # Масштабируем под доступную область с сохранением пропорций
            target_size = self.image_label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
            
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(False)  # Управляем масштабом вручную
            
            # Центрируем изображение
            self.image_label.setAlignment(Qt.AlignCenter)
            
            # Сохраняем путь к текущему изображению
            self.current_image_path = image_path
            
        except Exception as e:
            self.logs_text.append(f"Ошибка отображения изображения: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            self.image_label.setText("Предварительный просмотр сгенерированного изображения\n\nОшибка загрузки изображения")

    def _run_xai_for_image(self, image_path: str, class_name: str):
        """Запуск лёгкого XAI-оверлея для конкретного изображения без открытия окна."""
        try:
            # Не открываем изображение, просто сохраняем оверлей рядом в xai_results
            from xai import run_xai_analysis as _xai_func
        except Exception:
            # Пытаемся путь-импорт
            try:
                import importlib.util, os as _os
                p = _os.path.join(os.getcwd(), 'xai', 'xai_integration.py')
                if not os.path.exists(p):
                    p = _os.path.join(os.getcwd(), 'XAI', 'xai_integration.py')
                spec = importlib.util.spec_from_file_location('xai_xai_integration_dynamic', p)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                _xai_func = getattr(mod, 'run_xai_analysis', None)
            except Exception as e:
                self.logs_text.append(f"Импорт XAI hook не удался: {str(e)}")
                return
        try:
            device = getattr(self.generator, 'device', None)
            classifier_path = os.path.join(os.getcwd(), 'checkpoints', 'classifier.pth')
            save_dir = os.path.join(os.getcwd(), 'xai_results')
            _, saved_path = _xai_func(
                image_path,
                device=device,
                classifier_path=classifier_path,
                save_dir=save_dir
            )
            self.logs_text.append(f"XAI (периодический) сохранен: {saved_path}")
        except Exception as e:
            self.logs_text.append(f"Ошибка XAI hook: {str(e)}")
    
    def on_xai_toggle(self, checked):
        """Обработчик переключения XAI Mode"""
        self.xai_mode = bool(checked)
        self.logs_text.append(f"XAI Mode: {'ON' if self.xai_mode else 'OFF'}")
        # Перекрашиваем кнопку при включении
        try:
            if self.xai_mode:
                # Светло-серый фон в ON
                self.xai_mode_btn.setStyleSheet("QPushButton { background-color: #E0E0E0; border: 2px solid #A0A0A0; font-weight: bold; }")
            else:
                # Сброс к стилю по умолчанию
                self.xai_mode_btn.setStyleSheet("")
        except Exception:
            pass
        # Перерисовываем текущее изображение при смене режима, если оно уже показано
        try:
            if self.current_image_path and os.path.exists(self.current_image_path):
                self.display_image(self.current_image_path)
        except Exception as e:
            self.logs_text.append(f"Ошибка обновления изображения: {str(e)}")
        
        # Обновляем частоту шагов для XAI пайплайна через переменную окружения
        try:
            os.environ["XAI_SAVE_EVERY_N"] = str(self.xai_step_spin.value())
            self.logs_text.append(f"XAI n_steps set to: {self.xai_step_spin.value()}")
        except Exception:
            pass
            
    def open_xai_results_directory(self):
        """Открывает папку с результатами XAI"""
        try:
            # Создаем папку XAI если не существует
            xai_dir = os.path.join(os.getcwd(), "xai_results")
            if not os.path.exists(xai_dir):
                os.makedirs(xai_dir, exist_ok=True)
                
            # Открываем папку в проводнике Windows
            os.startfile(xai_dir)
            self.logs_text.append(f"Открыта папка результатов XAI: {xai_dir}")
            
        except Exception as e:
            self.logs_text.append(f"Ошибка открытия папки результатов XAI: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def open_checkpoints_directory(self):
        """Открывает папку с чекпоинтами"""
        try:
            if hasattr(self, 'selected_models_dir') and self.selected_models_dir:
                # Открываем папку в проводнике Windows
                os.startfile(self.selected_models_dir)
                self.logs_text.append(f"Открыта папка чекпоинтов: {self.selected_models_dir}")
            else:
                QMessageBox.information(self, "Информация", "Сначала выберите папку с моделями!")
                
        except Exception as e:
            self.logs_text.append(f"Ошибка открытия папки чекпоинтов: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_class_folder_clicked(self, item):
        """Обработчик клика по папке класса"""
        try:
            class_name = item.text()
            self.logs_text.append(f"Выбрана папка класса: {class_name}")
            self.load_images_from_class(class_name)
        except Exception as e:
            self.logs_text.append(f"Ошибка выбора папки класса: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_image_file_clicked(self, item):
        """Обработчик клика по файлу изображения"""
        try:
            filename = item.text()
            image_path = item.data(Qt.UserRole)  # Получаем полный путь к файлу
            
            if image_path and os.path.exists(image_path):
                self.display_image(image_path)
                self.logs_text.append(f"Отображается: {filename}")
            else:
                self.logs_text.append("Ошибка: Файл изображения не найден")
        except Exception as e:
            self.logs_text.append(f"Ошибка выбора файла изображения: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def load_images_from_class(self, class_name):
        """Загружает список изображений из папки класса"""
        try:
            if not hasattr(self, 'selected_output_dir') or not self.selected_output_dir:
                return
                
            class_dir = os.path.join(self.selected_output_dir, class_name)
            
            if not os.path.exists(class_dir):
                self.images_list.clear()
                return
                
            # Очищаем список изображений
            self.images_list.clear()
            
            # Ищем изображения в папке класса
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            found_images = []
            
            for file in os.listdir(class_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(class_dir, file)
                    found_images.append((file, image_path))
                    
            # Сортируем по имени файла
            found_images.sort(key=lambda x: x[0])
            
            # Добавляем в список
            for filename, image_path in found_images:
                item = QListWidgetItem(filename)
                item.setData(Qt.UserRole, image_path)  # Сохраняем полный путь
                self.images_list.addItem(item)
                
            self.logs_text.append(f"Загружено {len(found_images)} изображений из класса '{class_name}'")
            
        except Exception as e:
            self.logs_text.append(f"Ошибка загрузки изображений из класса: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def update_file_lists(self):
        """Обновляет списки папок классов и файлов"""
        try:
            if not hasattr(self, 'selected_output_dir') or not self.selected_output_dir:
                return
            
            # Очищаем списки
            self.class_folders_list.clear()
            self.images_list.clear()
            
            # Получаем список папок классов
            if os.path.exists(self.selected_output_dir):
                items = os.listdir(self.selected_output_dir)
                
                for item in items:
                    item_path = os.path.join(self.selected_output_dir, item)
                    if os.path.isdir(item_path):
                        self.class_folders_list.addItem(item)
                        
                self.logs_text.append(f"Найдено {self.class_folders_list.count()} папок классов")
            
        except Exception as e:
            self.logs_text.append(f"Ошибка обновления списков файлов: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_image_clicked(self, event):
        """Обработчик клика по изображению для открытия в полном размере"""
        try:
            if hasattr(self, 'current_image_path') and self.current_image_path:
                # Открываем изображение в стандартном приложении Windows
                os.startfile(self.current_image_path)
                self.logs_text.append(f"Открыто изображение: {self.current_image_path}")
            else:
                QMessageBox.information(self, "Информация", "Нет изображения для просмотра!")
                
        except Exception as e:
            self.logs_text.append(f"Ошибка открытия изображения: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def select_models_directory(self):
        """Выбор папки с моделями"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку с моделями", 
            os.getcwd(),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            # Проверяем, что это папка checkpoints
            if os.path.basename(directory) == "checkpoints":
                self.selected_models_dir = directory
                self.config_manager.update_path("checkpoints", directory)
                
                # Обновляем UI
                self.model_path_label.setText(f"Model Path: {directory}")
                self.logs_text.append(f"Выбрана папка с моделями: {directory}")
                
                # Проверяем доступные модели
                self.check_available_models()
                
                # Показываем изображения если они есть
                if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                    self.show_first_generated_image()
                    
                self.logs_text.append(f"Выбрана папка с моделями: {directory}")
                
            else:
                QMessageBox.warning(
                    self, 
                    "Неверная папка", 
                    "Пожалуйста, выберите папку 'checkpoints' с моделями"
                )
                
    def select_output_directory(self):
        """Выбор папки для вывода"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Выберите папку для сохранения изображений", 
            os.getcwd(),
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.selected_output_dir = directory
            self.config_manager.update_path("output", directory)
            
            # Обновляем UI
            self.logs_text.append(f"💾 Выбрана папка для вывода: {directory}")
            
            # Обновляем дерево проекта
            self.update_project_tree()
            
            # Обновляем списки файлов
            self.update_file_lists()
            
            # Показываем изображения если они есть
            if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                self.show_first_generated_image()
                
            self.logs_text.append(f"Выбрана папка для вывода: {directory}")
            
    def check_available_models(self):
        """Проверяет доступные модели в выбранной папке"""
        if not self.selected_models_dir:
            return
            
        try:
            # Обновляем доступные классы
            self.available_classes = self.generator.get_available_classes()
            
            # Обновляем UI для каждого класса
            for class_name, widgets in self.class_widgets.items():
                is_available = class_name in self.available_classes
                widgets['checkbox'].setEnabled(is_available)
                widgets['spinbox'].setEnabled(is_available)
                
                if not is_available:
                    widgets['checkbox'].setToolTip(f"Модель для класса {class_name} недоступна")
                else:
                    widgets['checkbox'].setToolTip("")
            
            # Обновляем информацию
            self.available_models_label.setText(f"Available Models: {len(self.available_classes)}")
            self.logs_text.append(f"Найдено {len(self.available_classes)} доступных моделей")

            
        except Exception as e:
            self.logs_text.append(f"ОШИБКА: Проверка модели не удалась: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def update_project_tree(self):
        """Обновляет дерево проекта"""
        if self.selected_output_dir:
            # Обновляем элемент generated_images
            root = self.project_tree.topLevelItem(0)
            for i in range(root.childCount()):
                child = root.child(i)
                if "generated_images" in child.text(0):
                    child.setText(0, f"generated_images ({self.selected_output_dir})")
                    break
                    
            self.logs_text.append(f"Дерево проекта обновлено для: {self.selected_output_dir}")
                    
    def on_device_changed(self, device_text):
        """Обработчик изменения устройства"""
        try:
            # Извлекаем название устройства из текста
            if "CUDA:" in device_text:
                device_name = device_text.split(" ")[0]  # Берем только "CUDA:0"
            else:
                device_name = device_text
                
            # Обновляем устройство в генераторе
            if hasattr(self, 'generator') and self.generator:
                # Создаем новое устройство
                if device_name == "CPU":
                    new_device = torch.device("cpu")
                else:
                    new_device = torch.device(device_name)
                    
                # Обновляем устройство в генераторе
                self.generator.device = new_device
                
                # Очищаем загруженные модели для перезагрузки на новом устройстве
                if hasattr(self.generator, 'model_manager'):
                    self.generator.model_manager.cleanup()
                    
                self._log_message(f"Устройство изменено на: {device_name}")
                self._log_message(f"Модели будут перезагружены при следующей генерации")
                
                # Обновляем UI
                self.device_info_label.setText(f"Device: {device_name}")
                
                # Обновляем информацию о памяти
                self.update_memory_info()
                
                # Показываем изображения если они есть
                if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                    self.show_first_generated_image()
                    
                self.logs_text.append(f"Device changed to: {device_name}")
                
        except Exception as e:
            self._log_message(f"Error changing device: {str(e)}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось изменить устройство: {str(e)}")
            import traceback
            self._log_message(f"Traceback: {traceback.format_exc()}")
            
    def on_start_clicked(self):
        """Обработчик нажатия кнопки Start"""
        if self.is_generating:
            return
            
        # Проверяем, что выбраны папки
        if not self.selected_models_dir:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите папку с моделями!")
            return
            
        if not self.selected_output_dir:
            QMessageBox.warning(self, "Предупреждение", "Сначала выберите папку для вывода!")
            return
            
        # Собираем конфигурацию классов
        class_configs = []
        for class_name, widgets in self.class_widgets.items():
            if widgets['checkbox'].isChecked() and widgets['available']:
                count = widgets['spinbox'].value()
                class_configs.append((class_name, count))
                
        if not class_configs:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один класс для генерации!")
            return
            
        # Запускаем генерацию
        self.start_generation(class_configs)
        

        
    def on_stop_clicked(self):
        """Обработчик нажатия кнопки Stop"""
        try:
            # Останавливаем генерацию мягко
            if self.generator:
                self.generator.stop_generation()
            # Ждём воркер, если он запущен
            if self.generation_worker and self.generation_worker.isRunning():
                self.generation_worker.wait(200)
                # Если не остановился, прерываем
                if self.generation_worker.isRunning():
                    self.generation_worker.terminate()
                    self.generation_worker.wait()
            # Останавливаем XAI пайплайн, если идёт
            if hasattr(self, 'xai_worker') and self.xai_worker and self.xai_worker.isRunning():
                try:
                    self.xai_worker.stop()
                    self.xai_worker.wait(500)
                except Exception:
                    pass
            self.is_generating = False
            self.update_ui_state()
            self.logs_text.append("Генерация остановлена")
        except Exception as e:
            self.logs_text.append(f"Ошибка остановки: {str(e)}")
        

        
    def on_regenerate_clicked(self):
        """Обработчик нажатия кнопки Regenerate"""
        # Для регенерации используем те же настройки
        if hasattr(self, 'last_class_configs'):
            self.start_generation(self.last_class_configs)

        else:
            QMessageBox.information(self, "Информация", "Сначала запустите генерацию!")

            
    def start_generation(self, class_configs):
        """Запускает генерацию изображений"""
        self.is_generating = True
        self.last_class_configs = class_configs
        self.update_ui_state()
        
        # Создаем воркер для асинхронной генерации
        self.generation_worker = GenerationWorker(
            self.generator, 
            class_configs, 
            self.selected_output_dir
        )
        
        # Подключаем сигналы
        self.generation_worker.progress_updated.connect(self.update_progress)
        self.generation_worker.log_updated.connect(self.logs_text.append)
        self.generation_worker.generation_finished.connect(self.on_generation_finished)
        
        # Запускаем воркер
        self.generation_worker.start()
        
        self.logs_text.append(f"Начинаем генерацию: {len(class_configs)} классов")

        # Если включён XAI Mode — параллельно запускаем полный XAI пайплайн
        if getattr(self, 'xai_mode', False):
            try:
                self.xai_worker = XAIWorker(working_dir=os.getcwd())
                self.xai_worker.log_updated.connect(self.logs_text.append)
                self.xai_worker.finished.connect(self.on_xai_finished)
                self.logs_text.append("XAI: параллельный анализ запущен")
                self.xai_worker.start()
            except Exception as e:
                self.logs_text.append(f"XAI: не удалось запустить: {str(e)}")

        
    def update_progress(self, current, total, message):
        """Обновляет прогресс-бар"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            
        # Обновляем лог
        self.logs_text.append(message)
        

        
    def on_generation_finished(self, results):
        """Обработчик завершения генерации"""
        self.is_generating = False
        self.update_ui_state()
        
        if "error" in results:
            self.logs_text.append(f"ОШИБКА: Генерация не удалась: {results['error']}")
            QMessageBox.critical(self, "Ошибка", f"Генерация завершена с ошибкой: {results['error']}")
        else:
            self.logs_text.append(f"🎉 Генерация завершена успешно!")
            self.logs_text.append(f"📊 Сгенерировано: {results.get('total_generated', 0)} изображений")
            
            # Показываем результат
            QMessageBox.information(
                self, 
                "Успех", 
                f"Генерация завершена!\nСгенерировано: {results.get('total_generated', 0)} изображений"
            )
            
        # Сбрасываем прогресс-бар
        self.progress_bar.setValue(0)
        
        # Обновляем дерево проекта
        self.update_project_tree()
        

        
        # Показываем изображения если они есть
        if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
            self.show_first_generated_image()
            
        self.logs_text.append(f"Генерация завершена - обновляем списки файлов")
        self.update_xai_lists()

    def on_xai_finished(self, success: bool):
        try:
            if success:
                self.logs_text.append("XAI: результаты сохранены (см. xai_results и логи выше)")
                self.update_xai_lists()
            else:
                self.logs_text.append("XAI: завершен с ошибками; см. логи выше")
        except Exception:
            pass

    def update_xai_lists(self):
        try:
            base = os.path.join(os.getcwd(), 'xai_results')
            self.xai_runs_list.clear()
            self.xai_files_list.clear()
            if not os.path.exists(base):
                os.makedirs(base, exist_ok=True)
            runs = []
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if os.path.isdir(p):
                    runs.append((name, os.path.getmtime(p)))
            runs.sort(key=lambda x: x[1], reverse=True)
            for name, _ in runs:
                self.xai_runs_list.addItem(name)
            if runs:
                self.xai_runs_list.setCurrentRow(0)
                self.on_xai_run_clicked(self.xai_runs_list.item(0))
            self.logs_text.append(f"XAI: найдено {len(runs)} запусков анализа")
        except Exception as e:
            self.logs_text.append(f"XAI: ошибка обновления списка: {str(e)}")

    def on_xai_run_clicked(self, item):
        try:
            base = os.path.join(os.getcwd(), 'xai_results')
            run_dir = os.path.join(base, item.text())
            self.xai_files_list.clear()
            if os.path.isdir(run_dir):
                files = sorted(os.listdir(run_dir))
                for f in files:
                    self.xai_files_list.addItem(f)
        except Exception as e:
            self.logs_text.append(f"XAI: ошибка клика по запуску: {str(e)}")

    def on_xai_file_clicked(self, item):
        try:
            base = os.path.join(os.getcwd(), 'xai_results')
            run_item = self.xai_runs_list.currentItem()
            if not run_item:
                return
            run_dir = os.path.join(base, run_item.text())
            file_name = item.text()
            path = os.path.join(run_dir, file_name)
            lower = file_name.lower()
            if any(lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                self.display_image(path)
                self.logs_text.append(f"XAI изображение отображено: {file_name}")
            elif lower.endswith('.json'):
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                import json as _json
                pretty = _json.dumps(data, indent=2, ensure_ascii=False)
                self.show_text_dialog(f"JSON: {file_name}", pretty)
            elif lower.endswith('.pkl') or lower.endswith('.pickle'):
                import pickle
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                summary = self._summarize_pickle(obj)
                self.show_text_dialog(f"PKL: {file_name}", summary)
            else:
                self.logs_text.append(f"XAI: неподдерживаемый тип файла: {file_name}")
        except Exception as e:
            self.logs_text.append(f"XAI: ошибка открытия файла: {str(e)}")

    def _summarize_pickle(self, obj) -> str:
        try:
            if isinstance(obj, dict):
                keys = list(obj.keys())
                return f"Type: dict\nKeys ({len(keys)}):\n- " + "\n- ".join(map(str, keys))
            return f"Type: {type(obj)}\nStr: {str(obj)[:2000]}"
        except Exception as e:
            return f"PKL summary error: {str(e)}"

    def show_text_dialog(self, title: str, content: str):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            layout = QVBoxLayout(dlg)
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setText(content)
            btn = QPushButton("Закрыть")
            btn.clicked.connect(dlg.accept)
            layout.addWidget(txt)
            layout.addWidget(btn)
            dlg.resize(700, 500)
            dlg.exec_()
        except Exception as e:
            self.logs_text.append(f"Dialog error: {str(e)}")
        
    def update_ui_state(self):
        """Обновляет состояние UI"""
        # Кнопки управления
        self.start_btn.setEnabled(not self.is_generating)
        self.stop_btn.setEnabled(self.is_generating)
        self.regenerate_btn.setEnabled(not self.is_generating)
        
        # Кнопки выбора папок
        self.select_model_btn.setEnabled(not self.is_generating)
        self.select_output_btn.setEnabled(not self.is_generating)
        
        # ComboBox устройства
        self.device_combo.setEnabled(not self.is_generating)
        
        # Чекбоксы и спинбоксы классов
        for widgets in self.class_widgets.values():
            widgets['checkbox'].setEnabled(not self.is_generating and widgets['available'])
            widgets['spinbox'].setEnabled(not self.is_generating and widgets['available'])
            

            
    def closeEvent(self, event):
        """Обработчик закрытия приложения"""
        try:
            # Останавливаем таймер обновления памяти
            if hasattr(self, 'memory_update_timer'):
                self.memory_update_timer.stop()
                
            # Останавливаем генерацию если запущена
            if self.is_generating:
                self.on_stop_clicked()
                
            # Очищаем ресурсы
            if self.generator:
                self.generator.cleanup()
                
            if self.logger:
                self.logger.close()
                
        except Exception as e:
            print(f"Ошибка при закрытии: {e}")
            

        event.accept()

def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    
    # Устанавливаем шрифт по умолчанию
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Создаем и показываем главное окно
    window = SyntheticDataGenerator()
    window.show()
    
    print("Приложение успешно запущено")
    
    # Запускаем приложение
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
