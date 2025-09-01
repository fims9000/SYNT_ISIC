# Core Package - ISIC Generator Infrastructure.

Основной пакет инфраструктуры для генерации синтетических данных ISIC.

## 🏗️ Архитектура

### Структура пакетов

```
core/
├── __init__.py              # Основной пакет
├── config/                  # Управление конфигурацией
│   ├── __init__.py
│   └── config_manager.py    # ConfigManager - менеджер настроек
├── utils/                   # Утилиты
│   ├── __init__.py
│   ├── path_manager.py      # PathManager - управление путями
│   └── logger.py            # Logger - система логирования
├── cache/                   # Управление кэшем
│   ├── __init__.py
│   └── cache_manager.py     # CacheManager - кэширование моделей
└── generator/               # Генерация изображений
    ├── __init__.py
    ├── model_manager.py     # ModelManager - управление моделями
    └── image_generator.py   # ImageGenerator - основная логика
```

## 🔧 Основные компоненты

### 1. ConfigManager
- **Назначение**: Управление конфигурацией и путями
- **Функции**: 
  - Загрузка/сохранение настроек
  - Автоматическое определение путей для разных ОС
  - Управление параметрами генерации
  - Настройки UI и расширенных функций

### 2. PathManager
- **Назначение**: Управление путями и файлами
- **Функции**:
  - Работа с относительными и абсолютными путями
  - Автоматическое создание директорий
  - Поиск чекпоинтов моделей
  - Генерация имен файлов ISIC

### 3. Logger
- **Назначение**: Система логирования
- **Функции**:
  - Многоуровневое логирование (DEBUG, INFO, WARNING, ERROR)
  - Ротация логов
  - Интеграция с GUI
  - Специальные логгеры для разных задач

### 4. CacheManager
- **Назначение**: Управление кэшем
- **Функции**:
  - Кэширование моделей с хешированием
  - Управление временными файлами
  - Автоматическая очистка
  - Статистика использования

### 5. ModelManager
- **Назначение**: Управление ML-моделями
- **Функции**:
  - Загрузка/выгрузка моделей
  - Валидация архитектуры
  - Управление памятью
  - Создание schedulers

### 6. ImageGenerator
- **Назначение**: Основная логика генерации
- **Функции**:
  - Генерация одиночных изображений
  - Пакетная генерация
  - Постобработка цветов
  - Управление прогрессом
  - Экспорт в CSV

## 🚀 Использование

### Базовый пример

```python
from core import ConfigManager, ImageGenerator

# Создаем менеджер конфигурации
config = ConfigManager()

# Создаем генератор
generator = ImageGenerator(config)

# Генерируем изображения
class_configs = [("MEL", 5), ("NV", 10)]
results = generator.generate_images(class_configs)

# Очищаем ресурсы
generator.cleanup()
```

### Расширенный пример

```python
from core import ConfigManager, ImageGenerator, Logger

# Настраиваем логирование
logger = Logger()
logger.setup_gui_handler(gui_text_widget)

# Создаем генератор с callback'ами
generator = ImageGenerator()

def progress_callback(current, total, message):
    progress_bar.setValue(int(current / total * 100))
    log_widget.append(message)

generator.set_progress_callback(progress_callback)
generator.set_log_callback(log_widget.append)

# Генерируем с настройками
results = generator.generate_images(
    class_configs=[("MEL", 5)],
    output_dir="./custom_output",
    postprocess=True
)
```

## ⚙️ Конфигурация

### Основные настройки

```json
{
  "paths": {
    "checkpoints": "checkpoints",
    "output": "generated_images",
    "cache": "core/cache",
    "logs": "core/logs"
  },
  "generation": {
    "image_size": 128,
    "train_timesteps": 1000,
    "inference_timesteps": 1000
  },
  "ui": {
    "theme": "light",
    "language": "ru"
  },
  "advanced": {
    "enable_color_postprocessing": true,
    "enable_xai": false
  }
}
```

### Автоматические пути

- **Windows**: `%APPDATA%/ISICGenerator/config.json`
- **macOS**: `~/Library/Application Support/ISICGenerator/config.json`
- **Linux**: `~/.config/ISICGenerator/config.json`

## 🧪 Тестирование

Запустите тесты инфраструктуры:

```bash
python test_infrastructure.py
```

Тесты проверяют:
- ✅ Работу всех компонентов
- ✅ Интеграцию между модулями
- ✅ Обработку ошибок
- ✅ Управление ресурсами

## 🔒 Безопасность

- Автоматическая очистка временных файлов
- Валидация входных данных
- Обработка исключений
- Логирование всех операций

## 📈 Производительность

- Кэширование моделей
- Управление памятью CUDA
- Асинхронная генерация
- Оптимизация путей

## 🆘 Troubleshooting

### Частые проблемы

1. **Модели не загружаются**
   - Проверьте наличие чекпоинтов в `checkpoints/`
   - Убедитесь в корректности имен файлов

2. **Ошибки CUDA**
   - Проверьте доступность GPU
   - Очистите CUDA кэш: `torch.cuda.empty_cache()`

3. **Проблемы с путями**
   - Запустите `test_infrastructure.py`
   - Проверьте права доступа к директориям

### Логи

Все логи сохраняются в:
- `core/logs/generator.log` - основные логи
- `core/logs/errors.log` - только ошибки
- Консоль - информационные сообщения

## 🔄 Обновления

Для обновления инфраструктуры:
1. Обновите зависимости: `pip install -r requirements.txt`
2. Запустите тесты: `python test_infrastructure.py`
3. Проверьте совместимость с существующими конфигурациями

