#!/usr/bin/env python3
"""
Консольная версия ISIC Diffusion Generator v3.0 для сервера
Полная функциональность генерации изображений без GUI
Адаптировано для сервера с базовой директорией ~/MaxYura
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import List, Optional

# Рабочие импорты как в generate_test.py
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import csv
import re

print("✅ Все модули успешно импортированы")


class ConsoleGenerator:
    """Консольный генератор изображений ISIC для сервера"""
    
    def __init__(self):
        """Инициализация консольного генератора"""
        # Пути к моделям и данным для сервера (как в generate_test.py)
        self.base_dir = os.path.expanduser('~/MaxYura')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        self.output_dir = os.path.join(self.base_dir, 'generated_images')
        self.stats_path = os.path.join(self.base_dir, 'checkpoints', 'color_statistics.json')
        
        # Параметры генерации (как в generate_test.py)
        self.image_size = 128
        self.train_timesteps = 1000
        self.inference_timesteps = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Создаем директорию вывода
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Настройки для синтетического датасета ISIC2018
        self.synthetic_dir = os.path.join(self.base_dir, 'ISIC2018_Task3_synt')
        self.synthetic_csv = os.path.join(self.base_dir, 'ISIC2018_Task3_GroundTruth_synt.csv')
        self.last_isic_number = 34320  # Последний номер из исходного датасета
        
        # Создаем директорию для синтетических данных
        os.makedirs(self.synthetic_dir, exist_ok=True)
        
        # Загружаем цветовые статистики для постобработки
        self.color_stats = self._load_color_statistics()
        
        print(f"💻 Устройство: {self.device}")
        print(f"📁 Чекпоинты: {self.checkpoint_dir}")
        print(f"📁 Вывод: {self.output_dir}")
        print(f"📁 Синтетический датасет: {self.synthetic_dir}")
        print(f"📄 CSV метки: {self.synthetic_csv}")
        print(f"🔢 Последний номер ISIC: {self.last_isic_number}")
        if self.color_stats:
            print(f"🎨 Цветовые статистики: загружены ({len(self.color_stats)} классов)")
        else:
            print(f"⚠️  Цветовые статистики: не найдены")
    
    def _load_color_statistics(self):
        """Загружает цветовые статистики для постобработки"""
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Ошибка загрузки цветовых статистик: {e}")
        return None
    
    def _get_next_isic_number(self):
        """Получает следующий номер ISIC для синтетического датасета"""
        self.last_isic_number += 1
        return self.last_isic_number
    
    def _create_isic_filename(self, isic_number):
        """Создает имя файла в формате ISIC"""
        return f"ISIC_{isic_number:07d}.jpg"
    
    def _get_class_columns(self):
        """Возвращает список колонок классов для CSV"""
        return ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    def _create_csv_header(self):
        """Создает заголовок CSV файла"""
        columns = ['image'] + self._get_class_columns()
        return columns
    
    def _create_csv_row(self, image_name, class_name):
        """Создает строку CSV для изображения"""
        columns = self._get_class_columns()
        row = [image_name] + [0.0] * len(columns)
        
        # Устанавливаем 1.0 для соответствующего класса
        if class_name in columns:
            class_index = columns.index(class_name)
            row[class_index + 1] = 1.0
        
        return row
    
    def _initialize_synthetic_csv(self):
        """Инициализирует CSV файл с заголовком"""
        if not os.path.exists(self.synthetic_csv):
            with open(self.synthetic_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self._create_csv_header())
            print(f"📄 Создан новый CSV файл: {self.synthetic_csv}")
        else:
            print(f"📄 Используется существующий CSV файл: {self.synthetic_csv}")
    
    def _append_to_csv(self, image_name, class_name):
        """Добавляет запись в CSV файл"""
        try:
            with open(self.synthetic_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = self._create_csv_row(image_name, class_name)
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️  Ошибка записи в CSV: {e}")
    
    def _apply_color_postprocessing(self, image, class_name):
        """Применяет постобработку цветов на основе статистик"""
        if not self.color_stats or class_name not in self.color_stats:
            return image
        
        try:
            stats = self.color_stats[class_name]
            
            if "rgb" not in stats or "mean" not in stats["rgb"]:
                print(f"  ⚠️ Неполная статистика для класса {class_name}")
                return image
            
            # Получаем средние значения RGB для класса
            target_mean = stats["rgb"]["mean"]
            
            # Конвертируем изображение в numpy array если это PIL Image
            if hasattr(image, 'convert'):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Вычисляем текущие средние значения
            current_mean = np.mean(img_array, axis=(0, 1))
            
            # Вычисляем разницу для коррекции
            correction = np.array(target_mean) - current_mean
            
            # Применяем коррекцию
            corrected_array = np.clip(img_array + correction, 0, 255).astype(np.uint8)
            
            # Конвертируем обратно в PIL Image если исходное было PIL
            if hasattr(image, 'convert'):
                corrected_image = Image.fromarray(corrected_array)
                return corrected_image
            else:
                return corrected_array
            
        except Exception as e:
            print(f"⚠️  Ошибка постобработки для {class_name}: {e}")
            return image
    
    def _get_available_models(self):
        """Получает список доступных моделей"""
        models = {}
        if os.path.exists(self.checkpoint_dir):
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pth') and file.startswith('unet_') and file.endswith('_best.pth'):
                    class_name = file[5:-9]  # Извлекаем имя класса из имени файла
                    models[class_name] = os.path.join(self.checkpoint_dir, file)
        return models
    
    def get_available_classes(self):
        """Возвращает список доступных классов"""
        models = self._get_available_models()
        return list(models.keys())
    
    def get_model_info(self, class_name):
        """Возвращает информацию о модели"""
        models = self._get_available_models()
        if class_name in models:
            return {
                'class_name': class_name,
                'checkpoint_path': models[class_name],
                'device': str(self.device)
            }
        return None
    
    def show_available_classes(self):
        """Показывает доступные классы для генерации"""
        models = self._get_available_models()
        if not models:
            print("❌ Нет доступных моделей")
            return
        
        print("\n📋 Доступные классы для генерации:")
        print("=" * 40)
        for i, (class_name, checkpoint_path) in enumerate(models.items(), 1):
            print(f"{i:2d}. {class_name:8s} - {checkpoint_path}")
        print("=" * 40)
        print(f"Всего доступно классов: {len(models)}")
    
    def generate_single_image(self, class_name: str, output_path: str, postprocess: bool = True):
        """Генерирует одно изображение для указанного класса"""
        models = self._get_available_models()
        if class_name not in models:
            raise ValueError(f"Класс {class_name} недоступен")
        
        checkpoint_path = models[class_name]
        print(f"🎨 Загружаю модель для класса {class_name}: {checkpoint_path}")
        
        # Идентичная архитектура с обучением (как в generate_test.py)
        model = UNet2DModel(
            sample_size=self.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # Attention блок
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",  # Attention блок
                "UpBlock2D",
                "UpBlock2D"
            ),
            class_embed_type=None,
        ).to(self.device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()

        # Настройка scheduler идентичная обучению (как в generate_test.py)
        scheduler = DDPMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        
        scheduler.set_timesteps(self.inference_timesteps, device=self.device)

        with torch.no_grad():
            # Генерируем случайный шум
            sample = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

            # Процесс генерации (как в generate_test.py)
            for t in tqdm(scheduler.timesteps, desc=f"Генерация {class_name}"):
                noise_pred = model(sample, t).sample
                sample = scheduler.step(noise_pred, t, sample).prev_sample

            # Обратное преобразование нормализации
            image = sample.clamp(-1, 1)
            image = (image + 1) * 0.5
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)

            # Применяем постобработку цветов если включена
            if postprocess:
                print(f"  🔧 Применяю постобработку цветов для {class_name}")
                image = self._apply_color_postprocessing(image, class_name)

            # Сохраняем изображение в формате JPG (как в исходном датасете)
            img_pil = Image.fromarray(image)
            img_pil.save(output_path, 'JPEG', quality=95)
            
            # Очищаем память
            del sample, image, img_pil
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def generate_images(self, class_names: List[str], count: int, 
                       output_dir: Optional[str] = None, postprocess: bool = True):
        """
        Генерирует изображения для указанных классов
        
        Args:
            class_names: Список классов для генерации
            count: Количество изображений для каждого класса
            output_dir: Директория для сохранения (по умолчанию - стандартная)
            postprocess: Применять ли постобработку цветов
        """
        # Конвертируем в формат конфигураций для совместимости
        class_configs = [(class_name, count) for class_name in class_names]
        self.generate_images_with_configs(class_configs, output_dir, postprocess)
    
    def run_interactive(self):
        """Запускает интерактивный режим"""
        print("🎮 Интерактивный режим ISIC Diffusion Generator")
        print("=" * 50)
        
        while True:
            print("\nВыберите действие:")
            print("1. Показать доступные классы")
            print("2. Сгенерировать изображения")
            print("3. Выход")
            
            choice = input("\nВведите номер (1-3): ").strip()
            
            if choice == "1":
                self.show_available_classes()
                
            elif choice == "2":
                # Показываем доступные классы
                self.show_available_classes()
                
                # Пошаговый ввод классов и количества изображений
                class_configs = self._get_class_configs_interactive()
                
                if not class_configs:
                    print("❌ Не указаны классы для генерации")
                    continue
                
                # Запрашиваем директорию вывода
                output_dir = input("Директория вывода (Enter для стандартной): ").strip()
                if not output_dir:
                    output_dir = None
                
                # Запрашиваем постобработку
                postprocess_input = input("Применять постобработку цветов? (y/n, по умолчанию y): ").strip().lower()
                postprocess = postprocess_input != 'n'
                
                # Запускаем генерацию
                self.generate_images_with_configs(class_configs, output_dir, postprocess)
                
            elif choice == "3":
                print("👋 До свидания!")
                break
                
            else:
                print("❌ Некорректный выбор")
    
    def _get_class_configs_interactive(self):
        """Интерактивный ввод классов и количества изображений"""
        class_configs = []
        available_classes = self.get_available_classes()
        
        print(f"\n📝 Пошаговый ввод классов и количества изображений")
        print(f"💡 Доступные классы: {', '.join(available_classes)}")
        print(f"💡 Введите 'start' для запуска генерации или 'cancel' для отмены")
        print("=" * 60)
        
        while True:
            print(f"\n📊 Текущий список: {len(class_configs)} классов")
            if class_configs:
                for i, (cls, count) in enumerate(class_configs, 1):
                    print(f"  {i}. {cls}: {count} изображений")
            
            # Запрашиваем класс
            class_input = input(f"\nВведите класс (или 'start'/'cancel'): ").strip()
            
            if class_input.lower() == 'start':
                if not class_configs:
                    print("❌ Список классов пуст. Добавьте хотя бы один класс.")
                    continue
                break
            elif class_input.lower() == 'cancel':
                print("❌ Ввод отменен")
                return []
            
            # Проверяем существование класса
            if class_input not in available_classes:
                print(f"❌ Класс '{class_input}' недоступен")
                print(f"💡 Доступные классы: {', '.join(available_classes)}")
                continue
            
            # Проверяем, не добавлен ли уже этот класс
            if any(cls == class_input for cls, _ in class_configs):
                print(f"⚠️  Класс '{class_input}' уже добавлен")
                continue
            
            # Запрашиваем количество изображений для класса
            while True:
                try:
                    count_input = input(f"Введите количество изображений для класса '{class_input}': ").strip()
                    count = int(count_input)
                    if count <= 0:
                        print("❌ Количество должно быть положительным")
                        continue
                    break
                except ValueError:
                    print("❌ Некорректное количество")
                    continue
            
            # Добавляем конфигурацию
            class_configs.append((class_input, count))
            print(f"✅ Добавлен класс '{class_input}' с {count} изображениями")
        
        print(f"\n🎯 Финальная конфигурация:")
        for i, (cls, count) in enumerate(class_configs, 1):
            print(f"  {i}. {cls}: {count} изображений")
        
        return class_configs
    
    def generate_images_with_configs(self, class_configs, output_dir=None, postprocess=True):
        """
        Генерирует изображения на основе конфигураций классов
        
        Args:
            class_configs: Список кортежей (class_name, count)
            output_dir: Директория для сохранения
            postprocess: Применять ли постобработку цветов
        """
        # Определяем директорию вывода
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(self.synthetic_dir)  # Используем синтетическую папку по умолчанию
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем CSV файл
        self._initialize_synthetic_csv()
        
        print(f"\n🚀 Начинаю генерацию изображений...")
        print(f"📁 Директория вывода: {output_path}")
        print(f"🔧 Постобработка: {'Включена' if postprocess else 'Отключена'}")
        print(f"💻 Устройство: {self.device}")
        print(f"📄 CSV файл: {self.synthetic_csv}")
        
        total_generated = 0
        
        for class_name, count in class_configs:
            if class_name not in self.get_available_classes():
                print(f"⚠️  Класс {class_name} недоступен, пропускаю")
                continue
            
            print(f"\n🎨 Генерирую {count} изображений для класса: {class_name}")
            
            try:
                # Генерируем изображения в единой папке с правильной нумерацией ISIC
                for i in range(count):
                    # Получаем следующий номер ISIC
                    isic_number = self._get_next_isic_number()
                    isic_filename = self._create_isic_filename(isic_number)
                    output_file = output_path / isic_filename
                    
                    print(f"  Генерирую {i + 1}/{count}: {isic_filename}")
                    
                    self.generate_single_image(class_name, str(output_file), postprocess)
                    
                    # Добавляем запись в CSV
                    self._append_to_csv(isic_filename, class_name)
                    
                    total_generated += 1
                
                print(f"✅ Сгенерировано {count} изображений для {class_name}")
                
            except Exception as e:
                print(f"❌ Ошибка генерации для класса {class_name}: {e}")
                continue
        
        print(f"\n🎉 Генерация завершена!")
        print(f"📊 Всего сгенерировано изображений: {total_generated}")
        print(f"📁 Результаты сохранены в: {output_path}")
        print(f"📄 CSV файл обновлен: {self.synthetic_csv}")
        print(f"🔢 Последний использованный номер ISIC: {self.last_isic_number}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="ISIC Diffusion Generator v3.0 - Консольная версия для сервера",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Старый формат (одинаковое количество для всех классов)
  python console_generator_server.py --classes MEL,NV --count 5
  python console_generator_server.py --classes all --count 10 --output ./my_images
  
  # Новый формат (разное количество для каждого класса)
  python console_generator_server.py --class-counts "MEL:50,BCC:120,NV:30"
  python console_generator_server.py --class-counts "MEL:25,NV:100" --output ./custom_images
  
  # Интерактивный режим
  python console_generator_server.py --interactive
        """
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        help='Классы для генерации (через запятую или "all" для всех)'
    )
    
    parser.add_argument(
        '--count', '-n',
        type=int,
        help='Количество изображений для каждого класса (если указан один класс)'
    )
    
    parser.add_argument(
        '--class-counts', '-cc',
        type=str,
        help='Конфигурация классов и количества (формат: "MEL:50,BCC:120,NV:30")'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Директория для сохранения изображений'
    )
    
    parser.add_argument(
        '--no-postprocess',
        action='store_true',
        help='Отключить постобработку изображений'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Запустить интерактивный режим'
    )
    
    parser.add_argument(
        '--list-classes', '-l',
        action='store_true',
        help='Показать доступные классы и выйти'
    )
    
    args = parser.parse_args()
    
    # Создаем генератор
    generator = ConsoleGenerator()
    
    if args.list_classes:
        generator.show_available_classes()
        return
    
    if args.interactive:
        generator.run_interactive()
        return
    
    # Проверяем обязательные аргументы
    if not args.classes and not args.class_counts:
        print("❌ Для неинтерактивного режима требуются аргументы --classes или --class-counts")
        print("💡 Используйте --help для справки или --interactive для интерактивного режима")
        return
    
    # Обрабатываем конфигурацию классов
    if args.class_counts:
        # Парсим конфигурацию вида "MEL:50,BCC:120,NV:30"
        try:
            class_configs = []
            for item in args.class_counts.split(','):
                if ':' in item:
                    class_name, count_str = item.split(':', 1)
                    class_name = class_name.strip()
                    count = int(count_str.strip())
                    if count <= 0:
                        raise ValueError(f"Количество для {class_name} должно быть положительным")
                    class_configs.append((class_name, count))
                else:
                    print(f"⚠️  Некорректный формат: {item}")
                    return
            
            if not class_configs:
                print("❌ Не удалось разобрать конфигурацию классов")
                return
            
            print("🎯 Конфигурация классов:")
            for cls, count in class_configs:
                print(f"  {cls}: {count} изображений")
            
            # Запускаем генерацию с конфигурациями
            generator.generate_images_with_configs(
                class_configs=class_configs,
                output_dir=args.output,
                postprocess=not args.no_postprocess
            )
            return
            
        except ValueError as e:
            print(f"❌ Ошибка в конфигурации классов: {e}")
            return
        except Exception as e:
            print(f"❌ Ошибка парсинга конфигурации: {e}")
            return
    
    # Старый формат: --classes + --count
    if not args.count:
        print("❌ Для формата --classes требуется указать --count")
        return
    
    # Определяем классы
    if args.classes.lower() == 'all':
        class_names = generator.get_available_classes()
    else:
        class_names = [c.strip() for c in args.classes.split(',') if c.strip()]
    
    if not class_names:
        print("❌ Не указаны корректные классы")
        return
    
    # Запускаем генерацию
    generator.generate_images(
        class_names=class_names,
        count=args.count,
        output_dir=args.output,
        postprocess=not args.no_postprocess
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Генерация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)






