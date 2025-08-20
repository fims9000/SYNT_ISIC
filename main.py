#!/usr/bin/env python3
"""
ISIC Synthetic Data Generator - GUI Interface
Интерфейс для генерации синтетических данных ISIC
"""

import sys
import os
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
            self.log_updated.emit(f"ERROR: Critical error: {str(e)}")
            self.generation_finished.emit({"error": str(e)})

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
        except Exception as e:
            QMessageBox.critical(None, "Ошибка инициализации", 
                               f"Не удалось инициализировать core компоненты: {str(e)}")
            sys.exit(1)
        
        # Состояние приложения
        self.is_generating = False
        self.generation_worker = None
        self.selected_models_dir = ""
        self.selected_output_dir = ""
        
        # Состояние изображений
        self.current_image_path = None
        
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
        self.logs_text.append("System initialized. Ready for generation.")
        self.logs_text.append(f"Available models: {len(self.available_classes)}")
        self.logs_text.append(f"Available classes: {', '.join(self.available_classes)}")
        
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
                self.memory_info_label.setText(f"Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            else:
                self.memory_info_label.setText("Memory: CPU mode")
                
        except Exception as e:
            self.memory_info_label.setText("Memory: Error")
            self.logs_text.append(f"Memory update error: {str(e)}")
        
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
            print(f"Log cleanup error: {e}")
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("ISIC Synthetic Data Generator")
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
        top_group = QGroupBox("System Controls")
        top_layout = QVBoxLayout(top_group)
        top_layout.setSpacing(25)
        
        # Строка с кнопками
        button_layout = QHBoxLayout()
        
        # Кнопка выбора модели
        self.select_model_btn = QPushButton("Select Model")
        self.select_model_btn.setToolTip("Выберите папку с моделями (checkpoints)")
        
        # Кнопка выбора директории вывода
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.setToolTip("Выберите папку для сохранения изображений")
        
        # Тумблер XAI Mode
        self.xai_mode_btn = QPushButton("XAI Mode")
        self.xai_mode_btn.setCheckable(True)
        self.xai_mode_btn.setToolTip("Включить режим объяснимого ИИ")
        
        # ComboBox для выбора устройства с автоматическим определением
        self.device_combo = QComboBox()
        self._populate_device_combo()
        self.device_combo.setToolTip("Выберите устройство для генерации")
        
        # Добавляем кнопки в layout
        button_layout.addWidget(self.select_model_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.select_output_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.xai_mode_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(QLabel("Device:"))
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
        left_group = QGroupBox("Class Selection & Configuration")
        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(18)  # Увеличиваем междустрочный интервал
        
        # Создаем чекбоксы и спинбоксы для классов
        self.class_widgets = {}
        
        # Добавляем заголовок для классов
        class_header = QLabel("Available Classes:")
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
            class_layout.addWidget(QLabel("Count:"))
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
        control_header = QLabel("Generation Controls:")
        control_header.setStyleSheet("font-weight: bold; color: #404040; margin-bottom: 8px;")
        left_layout.addWidget(control_header)
        left_layout.addSpacing(5)
        
        # Кнопки управления
        self.start_btn = QPushButton("Start Generation")
        self.stop_btn = QPushButton("Stop Generation")
        self.regenerate_btn = QPushButton("Regenerate")
        
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
        center_group = QGroupBox("Image Generation Preview")
        center_layout = QVBoxLayout(center_group)
        center_layout.setSpacing(18)
        
        # Placeholder для изображения
        self.image_label = QLabel("Generated Image Preview\n\nSelect class folder and image file from the right panel")
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
        progress_label = QLabel("Generation Progress:")
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
        right_group = QGroupBox("Project Structure")
        right_layout = QVBoxLayout(right_group)
        right_layout.setSpacing(18)
        
        # Дерево проекта
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabel("Project Components")
        
        # Заполняем дерево
        root_item = QTreeWidgetItem(self.project_tree, ["Synthetic Data Project"])
        self.generated_images_item = QTreeWidgetItem(root_item, ["generated_images"])
        self.xai_results_item = QTreeWidgetItem(root_item, ["xai_results"])
        self.checkpoints_item = QTreeWidgetItem(root_item, ["checkpoints"])
        
        # Подключаем обработчик кликов
        self.project_tree.itemClicked.connect(self.on_project_item_clicked)
        
        self.project_tree.expandAll()
        
        right_layout.addWidget(self.project_tree)
        
        # Список файлов изображений
        files_group = QGroupBox("Generated Images")
        files_layout = QVBoxLayout(files_group)
        
        # Список папок классов
        self.class_folders_list = QListWidget()
        self.class_folders_list.setMaximumHeight(100)
        self.class_folders_list.itemClicked.connect(self.on_class_folder_clicked)
        
        # Список файлов изображений
        self.images_list = QListWidget()
        self.images_list.setMaximumHeight(150)
        self.images_list.itemClicked.connect(self.on_image_file_clicked)
        
        files_layout.addWidget(QLabel("Class Folders:"))
        files_layout.addWidget(self.class_folders_list)
        files_layout.addWidget(QLabel("Image Files:"))
        files_layout.addWidget(self.images_list)
        
        right_layout.addWidget(files_group)
        
        # Устанавливаем фиксированную ширину для правой панели
        right_group.setFixedWidth(250)
        
        main_layout.addWidget(right_group, 1, 3, 2, 1)
        
    def create_bottom_panel(self, main_layout):
        """Создает нижнюю панель"""
        # Панель логов
        logs_group = QGroupBox("System Logs")
        logs_layout = QVBoxLayout(logs_group)
        logs_layout.setSpacing(18)
        
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMinimumHeight(150)  # Увеличиваем минимальную высоту
        self.logs_text.setMaximumHeight(250)  # Увеличиваем максимальную высоту
        
        # Теперь можем добавлять отладочные сообщения
        self.logs_text.append("UI initialized successfully")
        
        logs_layout.addWidget(self.logs_text)
        logs_layout.addSpacing(5)
        
        # Добавляем информацию о логах
        logs_info = QLabel("System logs and generation progress will appear here")
        logs_info.setStyleSheet("color: #606060; font-style: italic; font-size: 9pt;")
        logs_layout.addWidget(logs_info)
        
        # Панель конфигурации
        config_group = QGroupBox("System Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(12)  # Увеличиваем междустрочный интервал
        
        # Статическая информация
        config_header = QLabel("Current Configuration:")
        config_header.setStyleSheet("font-weight: bold; color: #404040; margin-bottom: 8px;")
        config_layout.addWidget(config_header)
        config_layout.addSpacing(5)
        
        self.device_info_label = QLabel("Device: CPU")
        self.model_path_label = QLabel("Model Path: Not selected")
        self.available_models_label = QLabel(f"Available Models: {len(self.available_classes)}")
        self.color_config_label = QLabel("Color Config: Loaded")
        self.memory_info_label = QLabel("Memory: Not available")
        
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
        
    def on_project_item_clicked(self, item, column):
        """Обработчик кликов по элементам дерева проекта"""
        try:
            item_text = item.text(0)
            
            if item_text == "generated_images":
                self.open_generated_images_directory()
            elif item_text == "xai_results":
                self.open_xai_results_directory()
            elif item_text == "checkpoints":
                self.open_checkpoints_directory()
                
        except Exception as e:
            self.logs_text.append(f"Error opening directory: {str(e)}")
            
    def open_generated_images_directory(self):
        """Открывает папку с сгенерированными изображениями и показывает первое изображение"""
        try:
            if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                # Открываем папку в проводнике Windows
                os.startfile(self.selected_output_dir)
                self.logs_text.append(f"Opened generated images directory: {self.selected_output_dir}")
                
                # Показываем первое найденное изображение в интерфейсе
                self.show_first_generated_image()
            else:
                QMessageBox.information(self, "Информация", "Сначала выберите папку для вывода!")
                
        except Exception as e:
            self.logs_text.append(f"Error opening generated images directory: {str(e)}")
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
                self.logs_text.append(f"Loaded {len(found_images)} images from output directory")
            else:
                self.image_label.setText("Generated Image Preview\n\nNo images found in output directory")
                
        except Exception as e:
            self.logs_text.append(f"Error showing first image: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def display_image(self, image_path):
        """Отображает изображение в интерфейсе в исходном размере"""
        try:
            from PIL import Image
            
            # Загружаем изображение
            pil_image = Image.open(image_path)
            
            # Конвертируем в QPixmap
            from PyQt5.QtGui import QPixmap
            import io
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            # Отображаем изображение в исходном размере
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(False)  # Не растягиваем изображение
            
            # Центрируем изображение
            self.image_label.setAlignment(Qt.AlignCenter)
            
            # Сохраняем путь к текущему изображению
            self.current_image_path = image_path
            
        except Exception as e:
            self.logs_text.append(f"Error displaying image: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            self.image_label.setText("Generated Image Preview\n\nError loading image")
            
    def open_xai_results_directory(self):
        """Открывает папку с результатами XAI"""
        try:
            # Создаем папку XAI если не существует
            xai_dir = os.path.join(os.getcwd(), "xai_results")
            if not os.path.exists(xai_dir):
                os.makedirs(xai_dir, exist_ok=True)
                
            # Открываем папку в проводнике Windows
            os.startfile(xai_dir)
            self.logs_text.append(f"Opened XAI results directory: {xai_dir}")
            
        except Exception as e:
            self.logs_text.append(f"Error opening XAI results directory: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def open_checkpoints_directory(self):
        """Открывает папку с чекпоинтами"""
        try:
            if hasattr(self, 'selected_models_dir') and self.selected_models_dir:
                # Открываем папку в проводнике Windows
                os.startfile(self.selected_models_dir)
                self.logs_text.append(f"Opened checkpoints directory: {self.selected_models_dir}")
            else:
                QMessageBox.information(self, "Информация", "Сначала выберите папку с моделями!")
                
        except Exception as e:
            self.logs_text.append(f"Error opening checkpoints directory: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_class_folder_clicked(self, item):
        """Обработчик клика по папке класса"""
        try:
            class_name = item.text()
            self.logs_text.append(f"Selected class folder: {class_name}")
            self.load_images_from_class(class_name)
        except Exception as e:
            self.logs_text.append(f"Error selecting class folder: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_image_file_clicked(self, item):
        """Обработчик клика по файлу изображения"""
        try:
            filename = item.text()
            image_path = item.data(Qt.UserRole)  # Получаем полный путь к файлу
            
            if image_path and os.path.exists(image_path):
                self.display_image(image_path)
                self.logs_text.append(f"Displaying: {filename}")
            else:
                self.logs_text.append("Error: Image file not found")
        except Exception as e:
            self.logs_text.append(f"Error selecting image file: {str(e)}")
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
                
            self.logs_text.append(f"Loaded {len(found_images)} images from class '{class_name}'")
            
        except Exception as e:
            self.logs_text.append(f"Error loading images from class: {str(e)}")
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
                        
                self.logs_text.append(f"Found {self.class_folders_list.count()} class folders")
            
        except Exception as e:
            self.logs_text.append(f"Error updating file lists: {str(e)}")
            import traceback
            self.logs_text.append(f"Traceback: {traceback.format_exc()}")
            
    def on_image_clicked(self, event):
        """Обработчик клика по изображению для открытия в полном размере"""
        try:
            if hasattr(self, 'current_image_path') and self.current_image_path:
                # Открываем изображение в стандартном приложении Windows
                os.startfile(self.current_image_path)
                self.logs_text.append(f"Opened image: {self.current_image_path}")
            else:
                QMessageBox.information(self, "Информация", "Нет изображения для просмотра!")
                
        except Exception as e:
            self.logs_text.append(f"Error opening image: {str(e)}")
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
                self.logs_text.append(f"Model directory selected: {directory}")
                
                # Проверяем доступные модели
                self.check_available_models()
                
                # Показываем изображения если они есть
                if hasattr(self, 'selected_output_dir') and self.selected_output_dir:
                    self.show_first_generated_image()
                    
                self.logs_text.append(f"Models directory selected: {directory}")
                
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
                
            self.logs_text.append(f"Output directory selected: {directory}")
            
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
            self.logs_text.append(f"Found {len(self.available_classes)} available models")

            
        except Exception as e:
            self.logs_text.append(f"ERROR: Model check failed: {str(e)}")
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
                    
            self.logs_text.append(f"Project tree updated for: {self.selected_output_dir}")
                    
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
                    
                self._log_message(f"Device changed to: {device_name}")
                self._log_message(f"Models will be reloaded on next generation")
                
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
            QMessageBox.warning(self, "Error", f"Не удалось изменить устройство: {str(e)}")
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
        if self.generation_worker and self.generation_worker.isRunning():
            self.generation_worker.terminate()
            self.generation_worker.wait()
            
        if self.generator:
            self.generator.stop_generation()
            
        self.is_generating = False
        self.update_ui_state()
        self.logs_text.append("Generation stopped")
        

        
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
        
        self.logs_text.append(f"Starting generation: {len(class_configs)} classes")

        
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
            self.logs_text.append(f"ERROR: Generation failed: {results['error']}")
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
            
        self.logs_text.append(f"Generation finished - updating file lists")
        
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
    
    print("Application started successfully")
    
    # Запускаем приложение
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
