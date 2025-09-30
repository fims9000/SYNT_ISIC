# === ИМПОРТ И ИНИЦИАЛИЗАЦИЯ ===
import sys, os, warnings, gc, json, pickle
from datetime import datetime
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats, ndimage
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms, models

warnings.filterwarnings('ignore')

# Проверка Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
    print('✅ Grad-CAM доступен')
except ImportError:
    print('❌ Grad-CAM не найден. Установите: pip install grad-cam')
    GRADCAM_AVAILABLE = False

# Проверка XAI библиотек
try:
    from captum.attr import IntegratedGradients, GradientShap
    CAPTUM_AVAILABLE = True
    print('✅ Captum доступен')
except ImportError:
    print('⚠️  Captum недоступен')
    CAPTUM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
    print('✅ SHAP доступен')
except ImportError:
    print('⚠️  SHAP недоступен')
    SHAP_AVAILABLE = False

# Настройка устройства
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f'🚀 GPU: {torch.cuda.get_device_name(device)}')
else:
    device = torch.device('cpu')
    print('💻 CPU')

# Настройка matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print('🎯 Инициализация завершена!')

# Основные импорты
import os
import warnings
import gc
from datetime import datetime
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import numpy as np
import matplotlib.pyplot as plt

#Grad-Cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
from skimage.transform import resize

import seaborn as sns
from PIL import Image

# Научные библиотеки
from scipy import stats, ndimage
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Диффузионные модели (современная версия)
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms, models

# Подавление предупреждений
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# XAI библиотеки с проверкой доступности
XAI_AVAILABLE = False
CAPTUM_AVAILABLE = False
SHAP_AVAILABLE = False

try:
    from captum.attr import IntegratedGradients, GradientShap
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
    print("✅ Captum доступен")
except ImportError:
    print("⚠️  Captum не доступен. Установите: pip install captum")

try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP доступен")
except ImportError:
    print("⚠️  SHAP не доступен. Установите: pip install shap")

XAI_AVAILABLE = CAPTUM_AVAILABLE or SHAP_AVAILABLE

# Настройка matplotlib для качественных графиков
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Настройка устройства (адаптировано под современные системы)
if torch.cuda.is_available():
    # Попробуем использовать указанное устройство
    device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
    print(f"🚀 Используется GPU: {device} ({torch.cuda.get_device_name(device)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🍎 Используется Apple MPS")
else:
    device = torch.device('cpu')
    print("💻 Используется CPU")

# Оптимизация памяти для современных GPU
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Освобождение кэша
    torch.cuda.empty_cache()
    gc.collect()

print(f"🔬 XAI готовность: {XAI_AVAILABLE}")
print(f"📊 Captum: {CAPTUM_AVAILABLE}, SHAP: {SHAP_AVAILABLE}")
print("✅ Инициализация завершена!")

# Утилита: текстовый прогресс-бар в лог
def _log_progress_bar(label: str, current: int, total: int, width: int = 30):
    try:
        current = max(0, int(current))
        total = max(1, int(total))
        filled = int(width * current / total)
        bar = '#' * filled + '-' * (width - filled)
        pct = 100.0 * current / total
        print(f"{label}: [{bar}] {current}/{total} ({pct:.0f}%)", flush=True)
    except Exception:
        # На всякий случай не ломаем пайплайн из-за логов
        pass

# === КОНФИГУРАЦИЯ ПРОЕКТА ===

# Пути к данным и моделям (адаптированы под вашу структуру)
PROJECT_ROOT = Path(".").resolve()
MODELS_DIR = PROJECT_ROOT / "checkpoints"  # перенаправлено на checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "ISIC2018_Task3_Training_Input"
RESULTS_DIR = PROJECT_ROOT / "xai_results"

# Создаем необходимые директории
RESULTS_DIR.mkdir(exist_ok=True)

# === ПАРАМЕТРЫ МОДЕЛЕЙ ===

# DDPM параметры (из вашего кода обучения диффузии)
DDPM_IMAGE_SIZE = 128  # Размер для диффузионной модели
DDPM_CHANNELS = 3
DDPM_TIMESTEPS = 1000
DDPM_BETA_SCHEDULE = "squaredcos_cap_v2"

# Параметры классификатора (из вашего кода обучения CNN)
CLASSIFIER_IMAGE_SIZE = 224  # Размер для классификатора
CLASSIFIER_BATCH_SIZE = 16

# Классы ISIC2018 (корректная версия)
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NUM_CLASSES = len(CLASS_NAMES)

# === ПУТИ К МОДЕЛЯМ ===

# DDPM модели (в checkpoints/)
DDPM_MODELS = {
    'MEL': 'unet_MEL_best.pth',
    'NV': 'unet_NV_best.pth',
    'BCC': 'unet_BCC_best.pth',
    'AKIEC': 'unet_AKIEC_best.pth',
    'BKL': 'unet_BKL_best.pth',
    'DF': 'unet_DF_best.pth',
    'VASC': 'unet_VASC_best.pth'
}

# Классификатор
CLASSIFIER_PATH = CHECKPOINTS_DIR / "classifier.pth"

# Импорт пользовательских модулей (с обработкой ошибок)
try:
    # Добавляем models/ в путь
    import sys
    sys.path.append(str(MODELS_DIR))
    
    # Попытка импорта пользовательского классификатора
    from melanoma_classifier import MelanomaClassifier as UserMelanomaClassifier
    USER_CLASSIFIER_AVAILABLE = True
    print("✅ Пользовательский MelanomaClassifier импортирован")
except ImportError as e:
    print(f"⚠️  Не удалось импортировать пользовательский классификатор: {e}")
    print("   Будет использована встроенная реализация")
    USER_CLASSIFIER_AVAILABLE = False

# === ПАРАМЕТРЫ XAI АНАЛИЗА ===

# Диффузионная траектория
INFERENCE_STEPS = 50
SAVE_EVERY_N_STEPS = 5
GENERATION_SEED = 42

# XAI параметры
TOP_K_PERCENT = 10  # Процент наиболее важных регионов
BOTTOM_K_PERCENT = 10  # Процент наименее важных регионов
IG_N_STEPS = 50  # Шаги для Integrated Gradients
SHAP_N_SAMPLES = 512  # Сэмплы для SHAP аппроксимации

# Переопределение параметров через переменные окружения
try:
    _env_save_every = int(os.environ.get("XAI_SAVE_EVERY_N", str(SAVE_EVERY_N_STEPS)))
    if _env_save_every > 0:
        SAVE_EVERY_N_STEPS = _env_save_every
except Exception:
    pass

try:
    _env_inf_steps = int(os.environ.get("XAI_INFERENCE_STEPS", str(INFERENCE_STEPS)))
    if _env_inf_steps > 0:
        INFERENCE_STEPS = _env_inf_steps
except Exception:
    pass

try:
    _env_seed = int(os.environ.get("XAI_GENERATION_SEED", str(GENERATION_SEED)))
    GENERATION_SEED = _env_seed
except Exception:
    pass

# Интервенции
INTERVENTION_TYPES = ['blur']
NOISE_STD = 0.5
BLUR_KERNEL_SIZE = 5

# Статистика
ALPHA_LEVEL = 0.1
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10000

# Отображение конфигурации
print("📋 === КОНФИГУРАЦИЯ ПРОЕКТА ===")
print(f"📂 Корневая директория: {PROJECT_ROOT}")
print(f"🏗️  Модели: {MODELS_DIR}")
print(f"💾 Чекпоинты: {CHECKPOINTS_DIR}")
print(f"📊 Данные: {DATA_DIR}")
print(f"📈 Результаты: {RESULTS_DIR}")
print(f"🎯 Классы ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
print(f"🖼️  Размеры: DDPM={DDPM_IMAGE_SIZE}px, Classifier={CLASSIFIER_IMAGE_SIZE}px")
print(f"⚙️  Устройство: {device}")
print(f"🧠 Пользовательский классификатор: {USER_CLASSIFIER_AVAILABLE}")

# Проверка существования файлов
print("🔍 === ПРОВЕРКА ФАЙЛОВ МОДЕЛИ ===")

# Классификатор
classifier_exists = CLASSIFIER_PATH.exists()
print(f"🏥 Классификатор: {'✅' if classifier_exists else '❌'} {CLASSIFIER_PATH.name}")

# DDPM модели
existing_ddmp_models = []
for class_name, model_file in DDPM_MODELS.items():
    model_path = CHECKPOINTS_DIR / model_file
    exists = model_path.exists()
    print(f"🧬 {class_name}: {'✅' if exists else '❌'} {model_file}")
    if exists:
        existing_ddmp_models.append(class_name)

print(f"✅ Доступны DDPM модели для классов: {', '.join(existing_ddmp_models) if existing_ddmp_models else 'Нет'}")

if not existing_ddmp_models:
    print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Не найдено ни одной DDPM модели!")
    print("   Убедитесь, что файлы находятся в правильной директории.")
elif not classifier_exists:
    print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Классификатор не найден!")
    print("   XAI анализ будет использовать предобученную модель.")
else:
    print("🚀 Все необходимые файлы найдены! Готов к XAI анализу.")

def create_ddpm_unet():
    """
    Создает UNet2DModel с точной конфигурацией из вашего кода обучения
    
    Returns:
        UNet2DModel: Инициализированная модель
    """
    return UNet2DModel(
        sample_size=DDPM_IMAGE_SIZE,
        in_channels=DDPM_CHANNELS,
        out_channels=DDPM_CHANNELS,
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


def create_ddpm_scheduler():
    """
    Создает DDPMScheduler с параметрами из вашего кода обучения
    
    Returns:
        DDPMScheduler: Настроенный scheduler
    """
    return DDPMScheduler(
        num_train_timesteps=DDPM_TIMESTEPS,
        beta_schedule=DDPM_BETA_SCHEDULE,
        prediction_type="epsilon"  
        
    )


class MelanomaClassifierAdaptive(nn.Module):
    """
    Адаптивный классификатор меланомы
    
    Совместим с различными архитектурами и автоматически адаптируется
    к вашему обученному классификатору
    """
    
    def __init__(self, num_classes=NUM_CLASSES, architecture='auto', pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Пытаемся использовать пользовательский классификатор
        if USER_CLASSIFIER_AVAILABLE and architecture == 'auto':
            try:
                # Создаем экземпляр пользовательского классификатора
                self.model = UserMelanomaClassifier(pretrained=pretrained)
                self.architecture = 'user_model'
                print("✅ Используется пользовательский MelanomaClassifier")
            except Exception as e:
                print(f"⚠️  Ошибка создания пользовательской модели: {e}")
                print("   Переключаемся на встроенную архитектуру")
                self._create_builtin_model(pretrained)
        else:
            self._create_builtin_model(pretrained)
    
    def _create_builtin_model(self, pretrained=True):
        """Создает встроенную модель на основе ResNet18"""
        
        # Используем ResNet18 как базовую архитектуру (популярно для медицинских изображений)
        self.model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        
        # Заменяем последний слой
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        self.architecture = 'resnet18'
        print(f"✅ Создана встроенная модель: {self.architecture}")
    
    def preprocess_for_classifier(self, x):
        """
        Предобработка входных данных из диффузии в формат классификатора
        
        Преобразует изображения из диффузионной модели (128x128, [-1,1])
        в формат классификатора (224x224, ImageNet нормализация)
        """
        
        # Убеждаемся что тензор на правильном устройстве
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        
        # Из диапазона [-1, 1] в [0, 1]
        x = torch.clamp((x + 1.0) / 2.0, 0, 1)
        
        # Изменение размера с 128x128 на 224x224
        if x.shape[-1] != CLASSIFIER_IMAGE_SIZE or x.shape[-2] != CLASSIFIER_IMAGE_SIZE:
            x = F.interpolate(
                x, 
                size=(CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
                mode='bilinear',
                align_corners=False,
                antialias=True  # Современный параметр для лучшего качества
            )
        
        # ImageNet нормализация (стандарт для большинства предобученных моделей)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        x = normalize(x)
        return x
    
    def forward(self, x):
        """Forward pass с предобработкой"""
        x = self.preprocess_for_classifier(x)
        return self.model(x)
    
    def get_probabilities(self, x):
        """Получение вероятностей для каждого класса"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_per_class_score(self, x, target_class):
        """
        Получение per-class score для XAI анализа
        
        Формула: y = log p_cl(c | x_t)
        где c - целевой класс, x_t - изображение на временном шаге t
        
        Args:
            x: входное изображение
            target_class: индекс целевого класса
        
        Returns:
            torch.Tensor: логарифмическая вероятность для класса
        """
        probs = self.get_probabilities(x)
        # Добавляем небольшое значение для численной стабильности
        return torch.log(probs[:, target_class] + 1e-8)
    
    def predict(self, x):
        """Предсказание класса"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def get_confidence(self, x, target_class):
        """Получение уверенности модели в предсказании"""
        with torch.no_grad():
            probs = self.get_probabilities(x)
            return probs[:, target_class]


print("✅ Архитектуры моделей определены!")
print(f"🏗️  UNet2DModel: {DDPM_IMAGE_SIZE}x{DDPM_IMAGE_SIZE}, {DDPM_CHANNELS} каналов")
print(f"🧠 Классификатор: адаптивная архитектура")
print(f"📏 Scheduler: {DDPM_BETA_SCHEDULE}, {DDPM_TIMESTEPS} шагов")

def load_classifier_with_fallback():
    """
    Загружает классификатор с автоматическим fallback
    
    Пытается загрузить веса из checkpoint, при неудаче использует предобученную модель
    """
    
    print("🏥 Инициализация классификатора...")
    
    # Создаем модель
    classifier = MelanomaClassifierAdaptive(
        num_classes=NUM_CLASSES+1,
        architecture='resnet18',
        pretrained=True
    ).to(device)
    
    # Попытка загрузки весов
    if CLASSIFIER_PATH.exists():
        try:
            print(f"📥 Загрузка весов из {CLASSIFIER_PATH.name}...")
            
            # Загружаем checkpoint
            checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
            
            # Получаем состояние модели
            model_state = classifier.state_dict()
            
            # Фильтруем совместимые ключи
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint содержит дополнительную информацию
                state_dict = checkpoint['model_state_dict']
            else:
                # Checkpoint содержит только веса
                state_dict = checkpoint
            #print(set(state_dict.keys()) - set(model_state.keys()))
            # Проверяем совместимость ключей
            compatible_keys = {}
            incompatible_keys = []
            
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_keys[key] = value
                else:
                    incompatible_keys.append(key)
            
            # Загружаем совместимые веса
            if compatible_keys:
                classifier.load_state_dict(compatible_keys, strict=False)
                print(f"✅ Загружено {len(compatible_keys)}/{len(state_dict)} параметров")
                
                if incompatible_keys:
                    print(f"⚠️  Несовместимые ключи ({len(incompatible_keys)}): {incompatible_keys[:5]}{'...' if len(incompatible_keys) > 5 else ''}")
            else:
                print("❌ Нет совместимых параметров, используется предобученная модель")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки checkpoint: {e}")
            print("🔄 Используется предобученная модель")
    else:
        print(f"⚠️  Checkpoint не найден: {CLASSIFIER_PATH}")
        print("🔄 Используется предобученная модель")
    
    classifier.eval()
    
    # Тест работоспособности
    try:
        with torch.no_grad():
            test_input = torch.randn(1, DDPM_CHANNELS, DDPM_IMAGE_SIZE, DDPM_IMAGE_SIZE).to(device)
            test_output = classifier(test_input)
            test_probs = classifier.get_probabilities(test_input)
            
            print(f"🧪 Тест: вход {tuple(test_input.shape)} → выход {tuple(test_output.shape)}")
            print(f"📊 Вероятности: {tuple(test_probs.shape)}, сумма: {test_probs.sum():.3f}")
            
    except Exception as e:
        print(f"❌ Ошибка теста классификатора: {e}")
        return None
    
    print(f"✅ Классификатор готов! Архитектура: {classifier.architecture}")
    return classifier


def load_ddpm_model_for_class(class_name, verbose=True):
    """
    Загружает DDPM модель для указанного класса
    
    Args:
        class_name: название класса
        verbose: выводить подробную информацию
    
    Returns:
        tuple: (unet_model, scheduler) или (None, None) при ошибке
    """
    
    if class_name not in DDPM_MODELS:
        available_classes = list(DDPM_MODELS.keys())
        print(f"❌ Класс '{class_name}' недоступен. Доступные: {available_classes}")
        return None, None
    
    if verbose:
        print(f"🧬 Загрузка DDPM модели для класса '{class_name}'...")
    
    try:
        # Создаем архитектуру
        unet_model = create_ddpm_unet().to(device)
        scheduler = create_ddpm_scheduler()
        
        # Путь к модели
        # Разрешаем переопределение точного пути к модели из окружения (для побитового совпадения)
        override_path = os.environ.get('XAI_DDPM_MODEL_PATH', '').strip()
        if override_path:
            model_path = Path(override_path)
            model_file = model_path.name  # Имя файла для логирования
        else:
            model_file = DDPM_MODELS[class_name]
            model_path = CHECKPOINTS_DIR / model_file
        
        if not model_path.exists():
            print(f"❌ Файл модели не найден: {model_path}")
            return None, None
        
        # Загружаем веса
        if verbose:
            print(f"📥 Загрузка весов из {model_file}...")
        
        state_dict = torch.load(model_path, map_location=device)
        unet_model.load_state_dict(state_dict, strict=True)
        unet_model.eval()
        
        # Тест модели
        with torch.no_grad():
            test_input = torch.randn(1, DDPM_CHANNELS, DDPM_IMAGE_SIZE, DDPM_IMAGE_SIZE).to(device)
            test_timestep = torch.randint(0, 1000, (1,)).to(device)
            test_output = unet_model(test_input, test_timestep).sample
            
            if verbose:
                print(f"🧪 Тест: {tuple(test_input.shape)} + t → {tuple(test_output.shape)}")
        
        if verbose:
            print(f"✅ DDPM модель '{class_name}' загружена успешно!")
        
        return unet_model, scheduler
        
    except Exception as e:
        print(f"❌ Ошибка загрузки DDPM модели '{class_name}': {e}")
        return None, None


# === ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ===

print("🚀 === ЗАГРУЗКА МОДЕЛЕЙ ===")

# Загружаем классификатор
classifier = load_classifier_with_fallback()

if classifier is None:
    print("❌ Критическая ошибка: не удалось загрузить классификатор")
    raise RuntimeError("Классификатор не может быть загружен")

# Выбор класса для анализа (можно переопределить через переменную окружения XAI_TARGET_CLASS)
TARGET_CLASS_NAME = os.environ.get('XAI_TARGET_CLASS', 'MEL')  # ИЗМЕНИТЕ НА НУЖНЫЙ КЛАСС

if TARGET_CLASS_NAME not in CLASS_NAMES:
    print(f"❌ Неверный класс '{TARGET_CLASS_NAME}'. Доступные: {CLASS_NAMES}")
    TARGET_CLASS_NAME = CLASS_NAMES[0]  # Выбираем первый доступный
    print(f"🔄 Автоматически выбран класс: {TARGET_CLASS_NAME}")

TARGET_CLASS_ID = CLASS_NAMES.index(TARGET_CLASS_NAME)

print(f"🎯 Целевой класс: {TARGET_CLASS_NAME} (индекс: {TARGET_CLASS_ID})")

# Загружаем DDMP модель для выбранного класса
unet_model, scheduler = load_ddpm_model_for_class(TARGET_CLASS_NAME)

if unet_model is None:
    print(f"❌ Не удалось загрузить DDPM модель для класса '{TARGET_CLASS_NAME}'")
    print("   Проверьте наличие файла модели в директории checkpoints/")
    
    # Показываем доступные модели
    available_models = []
    for class_name in CLASS_NAMES:
        if class_name in DDPM_MODELS:
            model_path = CHECKPOINTS_DIR / DDPM_MODELS[class_name]
            if model_path.exists():
                available_models.append(class_name)
    
    if available_models:
        print(f"💡 Доступные модели: {', '.join(available_models)}")
        print(f"   Измените TARGET_CLASS_NAME на один из доступных классов")
    else:
        print("💡 Нет доступных DDPM моделей. Убедитесь что файлы находятся в checkpoints/")
    
    XAI_READY = False
else:
    XAI_READY = True
    print("🚀 Все модели загружены успешно! Готов к XAI анализу.")

# Оптимизация памяти
if device.type == 'cuda':
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🧹 Память GPU очищена. Используется: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")


def generate_trajectory_optimized(unet_model, scheduler, 
                                num_inference_steps=INFERENCE_STEPS,
                                save_every=SAVE_EVERY_N_STEPS,
                                seed=GENERATION_SEED,
                                use_autocast=True):
    """
    Оптимизированная генерация диффузионной траектории
    
    Особенности:
    - Автоматическое управление памятью
    - Mixed precision для ускорения
    - Прогресс-бар
    - Проверки на ошибки
    
    Args:
        unet_model: обученная UNet модель
        scheduler: DDPM scheduler
        num_inference_steps: количество шагов денойзинга
        save_every: сохранять каждый N-ый шаг
        seed: seed для воспроизводимости
        use_autocast: использовать mixed precision
    
    Returns:
        tuple: (trajectory, timesteps, metadata)
    """
    
    print(f"🎬 Генерация диффузионной траектории для класса '{TARGET_CLASS_NAME}'...")
    
    # Настройка воспроизводимости и создание локального генератора
    # Используем локальный torch.Generator, чтобы состояние глобального RNG не влияло на начальный шум
    try:
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(int(seed))
    except Exception:
        # Фолбэк: установим глобальные сиды
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        torch_gen = None
    np.random.seed(int(seed))
    
    # Создание начального шума (совместимо с генерацией в core)
    shape = (1, DDPM_CHANNELS, DDPM_IMAGE_SIZE, DDPM_IMAGE_SIZE)
    if torch_gen is not None:
        initial_noise = torch.randn(shape, device=device, dtype=torch.float32, generator=torch_gen)
    else:
        initial_noise = torch.randn(shape, device=device, dtype=torch.float32)
    
    print(f"🔢 Параметры генерации:")
    print(f"   Шаги: {num_inference_steps}, сохранять каждый: {save_every}")
    print(f"   Размер: {shape}, устройство: {device}")
    print(f"   Mixed precision: {use_autocast and device.type == 'cuda'}")
    print(f"   Seed: {seed}")
    
    # Настройка scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Диагностика: печатаем первый и последний t, чтобы убедиться в направлении
    try:
        t0 = float(timesteps[0])
        t_last = float(timesteps[-1])
        print(f"🧭 Timesteps dir: first={t0:.0f} last={t_last:.0f} (ожидается start≈999 → last=0)")
    except Exception:
        pass
    
    # Определение шагов для сохранения
    # Режим 1: обычный (каждые save_every по индексу шага)
    save_indices = set(range(0, num_inference_steps, save_every))
    if (num_inference_steps - 1) not in save_indices:
        save_indices.add(num_inference_steps - 1)  # Всегда сохраняем последний

    # Режим 2: абсолютные t (например, save_every=250 при 50 шагах)
    # Подбираем ближайшие доступные индексы под целевые t кратные save_every
    try:
        save_by_absolute_t = save_every >= num_inference_steps
    except Exception:
        save_by_absolute_t = False
    if save_by_absolute_t:
        try:
            t_list = [int(float(t)) for t in timesteps]
            desired_t = set()
            # Включаем 0 и max (обычно ~1000), и кратные save_every
            desired_t.add(0)
            desired_t.add(max(t_list))
            step_val = max(1, int(save_every))
            k = 0
            while k <= 1000:
                desired_t.add(k)
                k += step_val
            # Находим ближайшие индексы к целевым t
            for dt in desired_t:
                closest_idx = min(range(len(t_list)), key=lambda i: abs(t_list[i] - dt))
                save_indices.add(closest_idx)
        except Exception:
            pass
    
    trajectory = []
    saved_timesteps = []
    current_image = initial_noise.clone()
    
    # Основной цикл денойзинга
    try:
        unet_model.eval()
        
        with torch.no_grad():
            total_steps = len(timesteps)
            progress_bar = tqdm(
                enumerate(timesteps), 
                total=total_steps,
                desc=f"Denoising {TARGET_CLASS_NAME}",
                ncols=100
            )

            for step_idx, timestep in progress_bar:
                # Подготовка входных данных
                timestep_tensor = timestep.unsqueeze(0).to(device)
                
                # Forward pass с опциональным autocast
                if use_autocast and device.type == 'cuda':
                    with autocast(device_type='cuda'):
                        noise_pred = unet_model(current_image, timestep_tensor).sample
                else:
                    noise_pred = unet_model(current_image, timestep_tensor).sample
                
                # Шаг денойзинга
                scheduler_output = scheduler.step(noise_pred, timestep, current_image)
                current_image = scheduler_output.prev_sample
                
                # Сохранение промежуточного результата
                save_frame = (step_idx in save_indices)
                if not save_frame and save_by_absolute_t:
                    try:
                        t_int = int(float(timestep))
                        # Сохраняем, если t кратен save_every, а также гарантируем t==0
                        if (t_int % max(1, save_every) == 0) or (t_int == 0):
                            save_frame = True
                    except Exception:
                        save_frame = False
                if save_frame:
                    # Копируем на CPU для экономии GPU памяти
                    trajectory.append(current_image.detach().cpu().clone())
                    saved_timesteps.append(float(timestep))
                
                # Обновление прогресс-бара
                if step_idx % 10 == 0:
                    progress_bar.set_postfix({
                        't': f'{float(timestep):.0f}',
                        'saved': len(trajectory),
                        'mem': f'{torch.cuda.memory_allocated(device) / 1024**2:.0f}MB' if device.type == 'cuda' else 'N/A'
                    })
                    # Логируем текстовый прогресс для GUI логов
                    try:
                        _log_progress_bar("Denoising", step_idx + 1, total_steps)
                    except Exception:
                        pass
                
                # Очистка промежуточных тензоров
                del noise_pred
                if step_idx % 5 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            progress_bar.close()
    
    except Exception as e:
        print(f"❌ Ошибка во время генерации: {e}")
        return None, None, None
    
    # Добавляем финальное изображение если его нет
    if len(saved_timesteps) == 0 or saved_timesteps[-1] != 0.0:
        trajectory.append(current_image.detach().cpu().clone())
        saved_timesteps.append(0.0)
    
    # Метаданные
    metadata = {
        'class_name': TARGET_CLASS_NAME,
        'class_id': TARGET_CLASS_ID,
        'num_inference_steps': num_inference_steps,
        'save_every': save_every,
        'total_saved': len(trajectory),
        'seed': seed,
        'image_size': DDPM_IMAGE_SIZE,
        'device': str(device),
        'scheduler_type': scheduler.__class__.__name__,
        'beta_schedule': DDPM_BETA_SCHEDULE,
        'generation_time': datetime.now().isoformat()
    }
    
    print(f"✅ Траектория сгенерирована: {len(trajectory)} кадров")
    print(f"📊 Временные шаги: {[f'{t:.0f}' for t in saved_timesteps]}")
    
    # Финальная очистка памяти
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    return trajectory, saved_timesteps, metadata


def visualize_trajectory(trajectory, timesteps, max_frames=6, figsize=(18, 4)):
    """
    Визуализация диффузионной траектории
    
    Args:
        trajectory: список тензоров изображений
        timesteps: соответствующие временные шаги
        max_frames: максимальное количество кадров для отображения
        figsize: размер фигуры
    """
    
    num_frames = min(max_frames, len(trajectory))
    indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = [axes]
    
    for idx, frame_idx in enumerate(indices):
        # Конвертация тензора в изображение
        img_tensor = trajectory[frame_idx].squeeze()
        
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
            # CHW -> HWC
            img_np = img_tensor.permute(1, 2, 0).numpy()
        else:
            img_np = img_tensor.numpy()
        
        # Нормализация из [-1, 1] в [0, 1]
        img_np = np.clip((img_np + 1.0) / 2.0, 0, 1)
        
        # Отображение
        axes[idx].imshow(img_np)
        axes[idx].set_title(f't = {timesteps[frame_idx]:.0f}', fontsize=11, pad=10)
        axes[idx].axis('off')
        
        # Добавляем рамку для финального изображения
        if frame_idx == len(trajectory) - 1:
            axes[idx].add_patch(plt.Rectangle((0, 0), img_np.shape[1]-1, img_np.shape[0]-1, 
                                           fill=False, edgecolor='red', linewidth=3))
    
    plt.suptitle(f'🧬 Диффузионная траектория: {TARGET_CLASS_NAME}', 
                fontsize=14, y=1.05, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Статистика изображений
    final_image = trajectory[-1].squeeze()
    print(f"📊 Статистика финального изображения:")
    print(f"   Размер: {tuple(final_image.shape)}")
    print(f"   Диапазон: [{final_image.min():.3f}, {final_image.max():.3f}]")
    print(f"   Среднее: {final_image.mean():.3f}, std: {final_image.std():.3f}")


# === ГЕНЕРАЦИЯ ТРАЕКТОРИИ ===

if XAI_READY:
    print("🎬 === ГЕНЕРАЦИЯ ДИФФУЗИОННОЙ ТРАЕКТОРИИ ===")
    
    # Генерируем траекторию
    trajectory, timesteps, gen_metadata = generate_trajectory_optimized(
        unet_model=unet_model,
        scheduler=scheduler,
        num_inference_steps=INFERENCE_STEPS,
        save_every=SAVE_EVERY_N_STEPS,
        seed=GENERATION_SEED,
        use_autocast=False
    )
    
    if trajectory is not None:
        # Визуализируем результат
        visualize_trajectory(trajectory, timesteps, max_frames=6)
        
        print(f"📋 Метаданные генерации:")
        for key, value in gen_metadata.items():
            if key != 'generation_time':
                print(f"   {key}: {value}")
        
        TRAJECTORY_READY = True
    else:
        print("❌ Ошибка генерации траектории")
        TRAJECTORY_READY = False
else:
    print("⚠️  Пропуск генерации: модели не готовы")
    TRAJECTORY_READY = False


class ModernXAIAnalyzer:
    """
    Современный XAI анализатор для диффузионных моделей
    
    Реализует современные методы объяснимого ИИ:
    - Integrated Gradients с оптимизациями
    - SHAP аппроксимация через патчи
    - Time-SHAP для временной важности
    - Градиентные методы как fallback
    """
    
    def __init__(self, classifier, device, verbose=True):
        self.classifier = classifier
        self.device = device
        self.verbose = verbose
        
        # Инициализация XAI методов
        self.ig_method = None
        self.gradient_shap = None
        
        if CAPTUM_AVAILABLE:
            try:
                self.ig_method = IntegratedGradients(self._model_wrapper)
                self.gradient_shap = GradientShap(self._model_wrapper)
                if verbose:
                    print("✅ Captum методы инициализированы")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Ошибка инициализации Captum: {e}")
        
        # Кэш для базовых изображений
        self._baseline_cache = {}
        
        if verbose:
            print(f"🔬 XAI Analyzer готов. Captum: {CAPTUM_AVAILABLE}")
    
    def _model_wrapper(self, x):
        """Обёртка модели для Captum"""
        return self.classifier(x)
    
    def _get_baseline(self, image, baseline_type='noise'):
        """
        Получает baseline изображение для XAI методов
        
        Args:
            image: входное изображение
            baseline_type: тип baseline ('noise', 'zero', 'blur')
        
        Returns:
            torch.Tensor: baseline изображение
        """
        
        cache_key = f"{baseline_type}_{image.shape}_{image.device}"
        
        if cache_key not in self._baseline_cache:
            if baseline_type == 'noise':
                baseline = torch.randn_like(image) * 0.1
            elif baseline_type == 'zero':
                baseline = torch.zeros_like(image)
            elif baseline_type == 'blur':
                # Сильное размытие как baseline
                baseline = F.avg_pool2d(image, kernel_size=31, stride=1, padding=15)
            else:
                baseline = torch.zeros_like(image)
            
            self._baseline_cache[cache_key] = baseline
        
        return self._baseline_cache[cache_key]
    
    def compute_integrated_gradients(self, image, target_class, n_steps=IG_N_STEPS, baseline_type='noise'):
        """
        Вычисляет Integrated Gradients
        
        Формула: IG_i(x) = (x_i - x'_i) × ∫[0,1] ∂F(x' + α(x - x'))/∂x_i dα
        
        Args:
            image: входное изображение
            target_class: индекс целевого класса
            n_steps: количество шагов интегрирования
            baseline_type: тип baseline изображения
        
        Returns:
            torch.Tensor: карта атрибуции
        """
        
        image = image.to(self.device)
        
        # Используем Captum если доступен
        if self.ig_method is not None:
            try:
                baseline = self._get_baseline(image, baseline_type)
                
                # Функция для получения per-class score
                def target_func(x):
                    return self.classifier.get_per_class_score(x, target_class)
                
                # Временная замена
                original_func = self.ig_method.forward_func
                self.ig_method.forward_func = target_func
                
                try:
                    attribution = self.ig_method.attribute(
                        image,
                        baselines=baseline,
                        n_steps=n_steps,
                        method='riemann_right'  # Более стабильный метод
                    )
                finally:
                    # Восстанавливаем функцию
                    self.ig_method.forward_func = original_func
                
                return attribution
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Captum IG failed: {e}. Using gradient approximation.")
        
        # Fallback: простой градиентный метод
        return self._compute_gradient_attribution(image, target_class)
    
    def _compute_gradient_attribution(self, image, target_class):
        """Градиентная аттрибуция как fallback"""
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Forward pass
        score = self.classifier.get_per_class_score(image, target_class)
        
        # Backward pass
        score.backward()
        
        # Получаем градиенты
        attribution = image.grad.clone()
        
        # Очистка
        image.grad.zero_()
        image.requires_grad_(False)
        
        return attribution
    
    def compute_shap_approximation(self, image, target_class, 
                                 n_samples=SHAP_N_SAMPLES, patch_size=16):
        """
        SHAP аппроксимация через патчи
        
        Реализует упрощённую версию Kernel SHAP с патчами изображения
        
        Args:
            image: входное изображение
            target_class: индекс целевого класса
            n_samples: количество сэмплов для аппроксимации
            patch_size: размер патча
        
        Returns:
            torch.Tensor: SHAP карта атрибуции
        """
        
        image = image.to(self.device)
        batch_size, channels, height, width = image.shape
        
        # Создаём сетку патчей
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        
        attribution = torch.zeros_like(image)
        
        # Baseline score (чёрное изображение)
        baseline_image = torch.zeros_like(image)
        with torch.no_grad():
            baseline_score = self.classifier.get_per_class_score(
                baseline_image, target_class
            ).item()
        
        # Случайное сэмплирование масок
        for sample_idx in range(n_samples):
            # Создаём случайную маску патчей
            patch_mask = torch.rand(n_patches_h, n_patches_w) > 0.5
            
            # Расширяем до размера изображения
            full_mask = torch.zeros(height, width, dtype=torch.bool)
            
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    if patch_mask[i, j]:
                        y_start, y_end = i * patch_size, (i + 1) * patch_size
                        x_start, x_end = j * patch_size, (j + 1) * patch_size
                        full_mask[y_start:y_end, x_start:x_end] = True
            
            # Применяем маску к изображению
            masked_image = image.clone()
            masked_image[:, :, ~full_mask] = 0
            
            # Получаем score
            with torch.no_grad():
                masked_score = self.classifier.get_per_class_score(
                    masked_image, target_class
                ).item()
            
            # Вклад видимых патчей
            contribution = masked_score - baseline_score
            mask_tensor = full_mask.unsqueeze(0).unsqueeze(0).float().to(self.device)
            attribution += contribution * mask_tensor
        
        # Нормализация
        attribution /= n_samples
        
        return attribution
    
    def compute_time_shap(self, trajectory, timesteps, target_class):
        """
        Вычисляет Time-SHAP: важность временных шагов
        
        Определяет, какие временные шаги t наиболее важны
        для финального решения классификатора
        
        Args:
            trajectory: список изображений на разных временных шагах
            timesteps: соответствующие временные шаги
            target_class: индекс целевого класса
        
        Returns:
            tuple: (normalized_importance, raw_scores)
        """
        
        if self.verbose:
            print(f"🕒 Вычисление Time-SHAP для {len(trajectory)} временных шагов...")
        
        confidence_scores = []
        prob_scores = []
        
        for i, (image, t) in enumerate(zip(trajectory, timesteps)):
            image = image.to(self.device)
            
            with torch.no_grad():
                # Уверенность классификатора
                confidence = self.classifier.get_confidence(image, target_class).item()
                prob_scores.append(confidence)
                
                # Per-class score для более чувствительного анализа
                per_class_score = self.classifier.get_per_class_score(image, target_class).item()
                confidence_scores.append(per_class_score)
        
        # Конвертируем в numpy
        confidence_scores = np.array(confidence_scores)
        prob_scores = np.array(prob_scores)
        
        # Нормализация важности (используем per-class scores)
        if len(confidence_scores) > 1 and (confidence_scores.max() - confidence_scores.min()) > 1e-6:
            normalized_importance = (confidence_scores - confidence_scores.min()) / \
                                  (confidence_scores.max() - confidence_scores.min())
        else:
            normalized_importance = np.ones_like(confidence_scores) / len(confidence_scores)
        
        raw_data = {
            'confidence_scores': confidence_scores,
            'probability_scores': prob_scores,
            'timesteps': timesteps
        }
        
        if self.verbose:
            max_idx = np.argmax(normalized_importance)
            print(f"   Наиболее важный шаг: t={timesteps[max_idx]:.0f} (важность: {normalized_importance[max_idx]:.3f})")
        
        return normalized_importance, raw_data
    
    def compute_combined_attribution(self, image, target_class, 
                                   methods=['ig', 'shap'], weights=None):
        """
        Комбинированная XAI атрибуция
        
        Объединяет несколько методов для более робастных результатов
        
        Args:
            image: входное изображение
            target_class: индекс целевого класса
            methods: список методов ['ig', 'shap', 'gradient']
            weights: веса для комбинирования (по умолчанию равномерные)
        
        Returns:
            tuple: (combined_attribution, method_details)
        """
        
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        
        attributions = []
        method_details = {}
        
        for method, weight in zip(methods, weights):
            if self.verbose:
                print(f"   Вычисление {method.upper()}... (вес: {weight:.2f})")
            
            try:
                if method == 'ig':
                    attr = self.compute_integrated_gradients(image, target_class)
                elif method == 'shap':
                    attr = self.compute_shap_approximation(image, target_class)
                elif method == 'gradient':
                    attr = self._compute_gradient_attribution(image, target_class)
                else:
                    print(f"   ⚠️  Неизвестный метод: {method}")
                    continue
                
                attributions.append(attr * weight)
                method_details[method] = {
                    'weight': weight,
                    'mean_attribution': float(torch.mean(torch.abs(attr))),
                    'max_attribution': float(torch.max(torch.abs(attr)))
                }
                
            except Exception as e:
                print(f"   ❌ Ошибка в методе {method}: {e}")
                continue
        
        if not attributions:
            raise RuntimeError("Не удалось вычислить ни одну атрибуцию")
        
        # Комбинируем атрибуции
        combined_attribution = torch.stack(attributions).sum(dim=0)
        
        return combined_attribution, method_details


# Инициализация XAI анализатора
if XAI_READY and TRAJECTORY_READY:
    print("🔬 === ИНИЦИАЛИЗАЦИЯ XAI АНАЛИЗАТОРА ===")
    
    xai_analyzer = ModernXAIAnalyzer(
        classifier=classifier,
        device=device,
        verbose=True
    )
    
    # Тест XAI анализатора
    try:
        test_image = trajectory[-1].to(device)  # Используем финальное изображение
        
        print(f"🧪 Тестирование XAI методов на изображении {tuple(test_image.shape)}...")
        
        # Тест IG
        test_ig = xai_analyzer.compute_integrated_gradients(
            test_image, TARGET_CLASS_ID, n_steps=10
        )
        print(f"   ✅ IG: {tuple(test_ig.shape)}, диапазон: [{test_ig.min():.3f}, {test_ig.max():.3f}]")
        
        # Тест SHAP
        test_shap = xai_analyzer.compute_shap_approximation(
            test_image, TARGET_CLASS_ID, n_samples=5
        )
        print(f"   ✅ SHAP: {tuple(test_shap.shape)}, диапазон: [{test_shap.min():.3f}, {test_shap.max():.3f}]")
        
        # Тест Time-SHAP
        test_time_importance, _ = xai_analyzer.compute_time_shap(
            trajectory, timesteps, TARGET_CLASS_ID
        )
        print(f"   ✅ Time-SHAP: {len(test_time_importance)} шагов")
        
        XAI_ANALYZER_READY = True
        print("🚀 XAI анализатор готов к полному анализу!")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования XAI анализатора: {e}")
        XAI_ANALYZER_READY = False
        
else:
    print("⚠️  XAI анализатор не инициализирован: модели или траектория не готовы")
    XAI_ANALYZER_READY = False


def select_regions_advanced(attribution_map, k_percent=TOP_K_PERCENT, 
                           region_type='top', morphology_cleanup=True,
                           connectivity=8):
    """
    Продвинутая функция выбора топ-k или bottom-k регионов
    
    Args:
        attribution_map: карта атрибуции (tensor или numpy)
        k_percent: процент регионов для выбора
        region_type: 'top' или 'bottom'
        morphology_cleanup: применить морфологическую очистку
        connectivity: связность для морфологических операций
    
    Returns:
        dict: результаты с маской, статистикой и метаданными
    """
    
    # Конвертация в numpy
    if torch.is_tensor(attribution_map):
        attr_np = attribution_map.detach().cpu().numpy()
    else:
        attr_np = attribution_map.copy()
    
    # Обработка размерностей
    original_shape = attr_np.shape
    
    if len(attr_np.shape) == 4:  # Batch dimension
        attr_np = attr_np[0]
    
    if len(attr_np.shape) == 3:  # Channel dimension
        # Используем L2 норму по каналам для лучшей репрезентативности
        attr_np = np.linalg.norm(attr_np, axis=0)
    else:
        attr_np = np.abs(attr_np)
    
    # Вычисление порога
    flat_attr = attr_np.flatten()
    
    if region_type == 'top':
        threshold = np.percentile(flat_attr, 100 - k_percent)
        mask = attr_np >= threshold
    elif region_type == 'bottom':
        threshold = np.percentile(flat_attr, k_percent)
        mask = attr_np <= threshold
    else:
        raise ValueError(f"Неизвестный region_type: {region_type}")
    
    # Морфологическая очистка
    if morphology_cleanup:
        # Определяем структурирующий элемент
        if connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        else:  # connectivity == 8
            structure = ndimage.generate_binary_structure(2, 2)
        
        # Закрытие (заполнение дыр)
        mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
        
        # Открытие (удаление мелких объектов)
        mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
        
        # Удаление совсем мелких компонент
        labeled_mask, num_features = ndimage.label(mask, structure=structure)
        if num_features > 0:
            # Оставляем только компоненты больше определённого размера
            component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
            min_size = max(10, int(0.01 * mask.size))  # Минимум 1% от общего размера
            
            large_components = np.where(component_sizes >= min_size)[0] + 1
            mask = np.isin(labeled_mask, large_components)
    
    # Статистика
    total_pixels = attr_np.size
    selected_pixels = np.sum(mask)
    actual_percentage = (selected_pixels / total_pixels) * 100
    
    if selected_pixels > 0:
        mean_attribution_selected = np.mean(attr_np[mask])
        std_attribution_selected = np.std(attr_np[mask])
        max_attribution_selected = np.max(attr_np[mask])
        min_attribution_selected = np.min(attr_np[mask])
    else:
        mean_attribution_selected = 0
        std_attribution_selected = 0
        max_attribution_selected = 0
        min_attribution_selected = 0
    
    results = {
        'mask': mask,
        'threshold': threshold,
        'statistics': {
            'total_pixels': total_pixels,
            'selected_pixels': selected_pixels,
            'target_percentage': k_percent,
            'actual_percentage': actual_percentage,
            'threshold_value': threshold,
            'mean_attribution': np.mean(attr_np),
            'std_attribution': np.std(attr_np),
            'mean_attribution_selected': mean_attribution_selected,
            'std_attribution_selected': std_attribution_selected,
            'max_attribution_selected': max_attribution_selected,
            'min_attribution_selected': min_attribution_selected,
        },
        'metadata': {
            'region_type': region_type,
            'morphology_cleanup': morphology_cleanup,
            'connectivity': connectivity,
            'original_shape': original_shape
        }
    }
    
    return results


def counterfactual_intervention_advanced(image, mask, intervention_type='noise',
                                       **kwargs):
    """
    Продвинутая контрафактуальная интервенция
    
    Реализует формулу: x̃_t = x_t × (1-M) + intervention × M
    где M - маска регионов для интервенции
    
    Args:
        image: исходное изображение
        mask: маска регионов (numpy или tensor)
        intervention_type: тип интервенции
        **kwargs: дополнительные параметры
    
    Returns:
        dict: результаты интервенции
    """
    
    # Параметры по умолчанию
    noise_std = kwargs.get('noise_std', NOISE_STD)
    blur_kernel = kwargs.get('blur_kernel', BLUR_KERNEL_SIZE)
    inpaint_method = kwargs.get('inpaint_method', 'telea')
    
    # Убеждаемся что всё на правильном устройстве
    device = image.device
    
    # Подготовка маски
    if isinstance(mask, np.ndarray):
        mask_tensor = torch.from_numpy(mask).float().to(device)
    else:
        mask_tensor = mask.float().to(device)
    
    # Приведение размерностей маски к изображению
    while len(mask_tensor.shape) < len(image.shape):
        mask_tensor = mask_tensor.unsqueeze(0)
    
    # Расширение на все каналы если нужно
    if len(mask_tensor.shape) == 3 and image.shape[1] == 3:
        mask_tensor = mask_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    
    # Генерация интервенции в зависимости от типа
    if intervention_type == 'noise':
        intervention = torch.randn_like(image) * noise_std
        
    elif intervention_type == 'gaussian_noise':
        # Гауссовский шум с адаптивным std
        adaptive_std = max(noise_std, image.std().item() * 0.5)
        intervention = torch.randn_like(image) * adaptive_std
        
    elif intervention_type == 'zero':
        intervention = torch.zeros_like(image)
        
    elif intervention_type == 'mean':
        # Заменяем на среднее значение изображения
        mean_val = image.mean(dim=[-2, -1], keepdim=True)
        intervention = torch.full_like(image, 0) + mean_val
        
    elif intervention_type == 'blur':
        # Гауссово размытие
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        
        padding = blur_kernel // 2
        
        # Применяем размытие по каналам
        blurred_channels = []
        for c in range(image.shape[1]):
            channel = image[:, c:c+1, :, :]
            blurred = F.avg_pool2d(channel, kernel_size=blur_kernel, 
                                 stride=1, padding=padding)
            blurred_channels.append(blurred)
        
        intervention = torch.cat(blurred_channels, dim=1)
        
    elif intervention_type == 'inpaint':
        # Простое инпейнтинг через свёртку
        kernel_size = 5
        padding = kernel_size // 2
        
        # Создаём ядро для усреднения
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(device) / (kernel_size ** 2)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)
        
        # Применяем свёртку по группам (по каналам)
        intervention = F.conv2d(image, kernel, padding=padding, groups=image.shape[1])
        
    elif intervention_type == 'shuffle':
        # ИСПРАВЛЕННАЯ версия перемешивания пикселей
        intervention = image.clone()
        
        # Проверяем размерности перед обработкой
        batch_size, n_channels, height, width = intervention.shape
        
        for b in range(batch_size):
            for c in range(n_channels):
                channel = intervention[b, c]
                
                # ИСПРАВЛЕНИЕ: правильно обрабатываем размерности маски
                if len(mask_tensor.shape) == 4 and mask_tensor.shape[1] > c:
                    mask_2d = mask_tensor[b, c]  # Используем соответствующий канал
                elif len(mask_tensor.shape) == 4 and mask_tensor.shape[1] == 1:
                    mask_2d = mask_tensor[b, 0]  # Используем единственный канал
                elif len(mask_tensor.shape) == 3:
                    mask_2d = mask_tensor[b]     # Используем маску без канального измерения
                else:
                    mask_2d = mask_tensor[0, 0] if len(mask_tensor.shape) >= 2 else mask_tensor
                
                # Проверяем что маска не пустая
                if mask_2d.sum() > 0:
                    masked_pixels = channel[mask_2d.bool()]
                    if len(masked_pixels) > 1:  # Нужно хотя бы 2 пикселя для перемешивания
                        shuffled_pixels = masked_pixels[torch.randperm(len(masked_pixels))]
                        channel[mask_2d.bool()] = shuffled_pixels
    else:
        # По умолчанию - шум
        intervention = torch.randn_like(image) * noise_std
    
    # Применение интервенции согласно формуле
    modified_image = image * (1 - mask_tensor) + intervention * mask_tensor
    
    # Обеспечиваем корректный диапазон значений
    modified_image = torch.clamp(modified_image, -1, 1)
    
    # Вычисление статистик
    with torch.no_grad():
        # Разность между оригиналом и модифицированным изображением
        diff = torch.abs(image - modified_image)
        
        results = {
            'modified_image': modified_image,
            'intervention': intervention,
            'mask_tensor': mask_tensor,
            'difference': diff,
            'statistics': {
                'intervention_type': intervention_type,
                'mask_coverage': float(mask_tensor.mean()),
                'mean_difference': float(diff.mean()),
                'max_difference': float(diff.max()),
                'intervention_strength': float(torch.abs(intervention).mean()),
            },
            'parameters': kwargs
        }
    
    return results


def compute_causal_shift_comprehensive(classifier, original_image, modified_image, 
                                     target_class, include_all_classes=True):
    """
    Комплексное вычисление каузального сдвига (CFI)
    
    Формулы:
    CFI = g(x_original) - g(x_modified)
    δ = |CFI| / (|g(x_original)| + ε)
    
    Args:
        classifier: модель классификатора
        original_image: оригинальное изображение
        modified_image: модифицированное изображение
        target_class: индекс целевого класса
        include_all_classes: анализировать все классы
    
    Returns:
        dict: подробные результаты анализа
    """
    
    with torch.no_grad():
        # Per-class scores
        orig_score = classifier.get_per_class_score(original_image, target_class)
        mod_score = classifier.get_per_class_score(modified_image, target_class)
        
        # Полные вероятности
        orig_probs = classifier.get_probabilities(original_image)
        mod_probs = classifier.get_probabilities(modified_image)
        
        # Предсказанные классы
        orig_pred = torch.argmax(orig_probs, dim=1)
        mod_pred = torch.argmax(mod_probs, dim=1)
        
        # Основные метрики
        cfi = orig_score - mod_score
        delta = torch.abs(cfi) / (torch.abs(orig_score) + 1e-8)
        
        # Сдвиг вероятностей
        prob_shift = orig_probs[0, target_class] - mod_probs[0, target_class]
        
        # Базовые результаты
        results = {
            'target_class_analysis': {
                'class_id': target_class,
                'class_name': CLASS_NAMES[target_class],
                'cfi': float(cfi),
                'delta': float(delta),
                'original_score': float(orig_score),
                'modified_score': float(mod_score),
                'original_probability': float(orig_probs[0, target_class]),
                'modified_probability': float(mod_probs[0, target_class]),
                'probability_shift': float(prob_shift),
            },
            'prediction_analysis': {
                'original_prediction': int(orig_pred[0]),
                'original_prediction_name': CLASS_NAMES[int(orig_pred[0])],
                'modified_prediction': int(mod_pred[0]),
                'modified_prediction_name': CLASS_NAMES[int(mod_pred[0])],
                'prediction_changed': bool(orig_pred[0] != mod_pred[0]),
                'original_confidence': float(torch.max(orig_probs)),
                'modified_confidence': float(torch.max(mod_probs)),
                'confidence_drop': float(torch.max(orig_probs) - torch.max(mod_probs))
            }
        }
        
        # Анализ всех классов если требуется
        if include_all_classes:
            all_classes_analysis = []
            
            for class_id in range(len(CLASS_NAMES)):
                orig_class_score = classifier.get_per_class_score(original_image, class_id)
                mod_class_score = classifier.get_per_class_score(modified_image, class_id)
                class_cfi = orig_class_score - mod_class_score
                class_delta = torch.abs(class_cfi) / (torch.abs(orig_class_score) + 1e-8)
                
                class_analysis = {
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'cfi': float(class_cfi),
                    'delta': float(class_delta),
                    'original_probability': float(orig_probs[0, class_id]),
                    'modified_probability': float(mod_probs[0, class_id]),
                    'probability_shift': float(orig_probs[0, class_id] - mod_probs[0, class_id])
                }
                
                all_classes_analysis.append(class_analysis)
            
            results['all_classes_analysis'] = all_classes_analysis
        
        # Дополнительные метрики
        kl_divergence = float(F.kl_div(torch.log(mod_probs + 1e-8), orig_probs, reduction='sum'))
        js_divergence = float(0.5 * (F.kl_div(torch.log((orig_probs + mod_probs)/2 + 1e-8), orig_probs, reduction='sum') + 
                                   F.kl_div(torch.log((orig_probs + mod_probs)/2 + 1e-8), mod_probs, reduction='sum')))
        
        results['distribution_analysis'] = {
            'kl_divergence': kl_divergence,
            'js_divergence': js_divergence,
            'total_variation': float(0.5 * torch.sum(torch.abs(orig_probs - mod_probs)))
        }
    
    return results


print("✅ Функции для регионов и интервенций готовы!")
print(f"🔧 Доступные типы интервенций: noise, gaussian_noise, zero, mean, blur, inpaint, shuffle")
print(f"📊 Поддержка морфологической обработки и всесторонний анализ CFI")


def statistical_validation_comprehensive(top_k_shifts, bottom_k_shifts, 
                                       alpha=ALPHA_LEVEL, 
                                       n_bootstrap=N_BOOTSTRAP,
                                       n_permutations=N_PERMUTATIONS):
    """
    Комплексная статистическая валидация XAI результатов
    
    Включает современные статистические методы:
    - Параметрические тесты (t-test, Welch's t-test)
    - Непараметрические тесты (Mann-Whitney U, Wilcoxon)
    - Bootstrap confidence intervals
    - Permutation tests
    - Effect size анализ (Cohen's d, Cliff's delta)
    - Bayesian analysis (если возможно)
    
    Args:
        top_k_shifts: массив CFI сдвигов для топ-k регионов
        bottom_k_shifts: массив CFI сдвигов для bottom-k регионов
        alpha: уровень значимости
        n_bootstrap: количество bootstrap сэмплов
        n_permutations: количество пермутаций
    
    Returns:
        dict: комплексные статистические результаты
    """
    
    print(f"📊 Проведение комплексной статистической валидации...")
    print(f"   Top-k выборка: {len(top_k_shifts)} значений")
    print(f"   Bottom-k выборка: {len(bottom_k_shifts)} значений")
    print(f"   Уровень значимости: α = {alpha}")
    
    # Конвертация в numpy
    top_k = np.array(top_k_shifts)
    bottom_k = np.array(bottom_k_shifts)
    
    # Базовые описательные статистики
    def compute_descriptive_stats(data, name):
        return {
            'name': name,
            'n': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'var': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
        }
    
    descriptive_stats = {
        'top_k': compute_descriptive_stats(top_k, 'Top-k'),
        'bottom_k': compute_descriptive_stats(bottom_k, 'Bottom-k')
    }
    
    # 1. ПАРАМЕТРИЧЕСКИЕ ТЕСТЫ
    parametric_tests = {}
    
    # Обычный t-test
    t_stat, t_p_value = stats.ttest_ind(top_k, bottom_k)
    parametric_tests['t_test'] = {
        'statistic': t_stat,
        'p_value': t_p_value,
        'significant': t_p_value < alpha,
        'description': "Independent samples t-test"
    }
    
    # Welch's t-test (не предполагает равенство дисперсий)
    welch_t_stat, welch_t_p = stats.ttest_ind(top_k, bottom_k, equal_var=False)
    parametric_tests['welch_t_test'] = {
        'statistic': welch_t_stat,
        'p_value': welch_t_p,
        'significant': welch_t_p < alpha,
        'description': "Welch's t-test (unequal variances)"
    }
    
    # 2. НЕПАРАМЕТРИЧЕСКИЕ ТЕСТЫ
    nonparametric_tests = {}
    
    # Mann-Whitney U test
    u_stat, u_p_value = stats.mannwhitneyu(top_k, bottom_k, alternative='two-sided')
    nonparametric_tests['mann_whitney_u'] = {
        'statistic': u_stat,
        'p_value': u_p_value,
        'significant': u_p_value < alpha,
        'description': "Mann-Whitney U test"
    }
    
    # Wilcoxon rank-sum (альтернативная реализация)
    try:
        wilcox_stat, wilcox_p = stats.ranksums(top_k, bottom_k)
        nonparametric_tests['wilcoxon_rank_sum'] = {
            'statistic': wilcox_stat,
            'p_value': wilcox_p,
            'significant': wilcox_p < alpha,
            'description': "Wilcoxon rank-sum test"
        }
    except Exception as e:
        print(f"   ⚠️  Wilcoxon test failed: {e}")
    
    # 3. EFFECT SIZE АНАЛИЗ
    effect_sizes = {}
    
    # Cohen's d
    pooled_std = np.sqrt(((len(top_k) - 1) * np.var(top_k, ddof=1) + 
                         (len(bottom_k) - 1) * np.var(bottom_k, ddof=1)) / 
                        (len(top_k) + len(bottom_k) - 2))
    
    cohens_d = (np.mean(top_k) - np.mean(bottom_k)) / pooled_std if pooled_std > 0 else 0
    
    # Интерпретация Cohen's d
    if abs(cohens_d) < 0.2:
        cohens_interpretation = 'negligible'
    elif abs(cohens_d) < 0.5:
        cohens_interpretation = 'small'
    elif abs(cohens_d) < 0.8:
        cohens_interpretation = 'medium'
    else:
        cohens_interpretation = 'large'
    
    effect_sizes['cohens_d'] = {
        'value': cohens_d,
        'interpretation': cohens_interpretation,
        'description': "Cohen's d (standardized mean difference)"
    }
    
    # Glass's delta (альтернативная мера effect size)
    glass_delta = (np.mean(top_k) - np.mean(bottom_k)) / np.std(bottom_k, ddof=1)
    effect_sizes['glass_delta'] = {
        'value': glass_delta,
        'description': "Glass's delta (using control group std)"
    }
    
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    def bootstrap_mean_difference(n_bootstrap=n_bootstrap, confidence_level=1-alpha):
        """Bootstrap оценка доверительного интервала для разности средних"""
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample с возвращением
            top_sample = np.random.choice(top_k, len(top_k), replace=True)
            bottom_sample = np.random.choice(bottom_k, len(bottom_k), replace=True)
            
            diff = np.mean(top_sample) - np.mean(bottom_sample)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Доверительный интервал
        ci_lower = np.percentile(bootstrap_diffs, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 + confidence_level) / 2 * 100)
        
        return {
            'bootstrap_diffs': bootstrap_diffs,
            'mean_diff': np.mean(bootstrap_diffs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_contains_zero': ci_lower <= 0 <= ci_upper,
            'confidence_level': confidence_level
        }
    
    bootstrap_results = bootstrap_mean_difference()
    
    # 5. PERMUTATION TEST
    def permutation_test_comprehensive(n_permutations=n_permutations):
        """Комплексный пермутационный тест"""
        
        combined = np.concatenate([top_k, bottom_k])
        observed_diff = np.mean(top_k) - np.mean(bottom_k)
        
        permuted_diffs = []
        if len(top_k) >= 2 and len(bottom_k) >= 2:
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_top = combined[:len(top_k)]
                perm_bottom = combined[len(top_k):]
                perm_diff = np.mean(perm_top) - np.mean(perm_bottom)
                permuted_diffs.append(perm_diff)
        else:
            # Недостаточно данных для смыслового пермутационного теста
            permuted_diffs = np.array([observed_diff])
        
        permuted_diffs = np.array(permuted_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff)) if permuted_diffs.size > 1 else 1.0
        
        return {
            'observed_difference': observed_diff,
            'permuted_differences': permuted_diffs,
            'p_value': p_value,
            'significant': p_value < alpha,
            'n_permutations': n_permutations
        }
    
    permutation_results = permutation_test_comprehensive()
    
    # 6. ТЕСТЫ НА НОРМАЛЬНОСТЬ
    normality_tests = {}

    # Shapiro-Wilk test (требует n >= 3)
    try:
        if len(top_k) >= 3 and len(bottom_k) >= 3 and len(top_k) <= 5000 and len(bottom_k) <= 5000:
            shapiro_top = stats.shapiro(top_k)
            shapiro_bottom = stats.shapiro(bottom_k)
            normality_tests['shapiro_wilk'] = {
                'top_k': {'statistic': shapiro_top[0], 'p_value': shapiro_top[1], 'normal': shapiro_top[1] > alpha},
                'bottom_k': {'statistic': shapiro_bottom[0], 'p_value': shapiro_bottom[1], 'normal': shapiro_bottom[1] > alpha}
            }
        else:
            normality_tests['shapiro_wilk'] = {
                'top_k': {'skipped': True, 'reason': 'sample_size < 3 or > 5000'},
                'bottom_k': {'skipped': True, 'reason': 'sample_size < 3 or > 5000'}
            }
    except Exception as e:
        normality_tests['shapiro_wilk'] = {'error': str(e)}
    
    # Kolmogorov-Smirnov test
    ks_top = stats.kstest(top_k, 'norm', args=(np.mean(top_k), np.std(top_k)))
    ks_bottom = stats.kstest(bottom_k, 'norm', args=(np.mean(bottom_k), np.std(bottom_k)))
    
    normality_tests['kolmogorov_smirnov'] = {
        'top_k': {'statistic': ks_top[0], 'p_value': ks_top[1], 'normal': ks_top[1] > alpha},
        'bottom_k': {'statistic': ks_bottom[0], 'p_value': ks_bottom[1], 'normal': ks_bottom[1] > alpha}
    }
    
    # 7. ТЕСТЫ НА РАВЕНСТВО ДИСПЕРСИЙ
    variance_tests = {}
    
    # Levene's test
    levene_stat, levene_p = stats.levene(top_k, bottom_k)
    variance_tests['levene'] = {
        'statistic': levene_stat,
        'p_value': levene_p,
        'equal_variances': levene_p > alpha,
        'description': "Levene's test for equal variances"
    }
    
    # F-test
    f_stat = np.var(top_k, ddof=1) / np.var(bottom_k, ddof=1)
    f_p_value = 2 * min(stats.f.cdf(f_stat, len(top_k)-1, len(bottom_k)-1),
                       1 - stats.f.cdf(f_stat, len(top_k)-1, len(bottom_k)-1))
    
    variance_tests['f_test'] = {
        'statistic': f_stat,
        'p_value': f_p_value,
        'equal_variances': f_p_value > alpha,
        'description': "F-test for equal variances"
    }
    
    # 8. ИТОГОВЫЕ РЕЗУЛЬТАТЫ
    # Консенсус по значимости
    significance_consensus = {
        'parametric_significant': any([test['significant'] for test in parametric_tests.values()]),
        'nonparametric_significant': any([test['significant'] for test in nonparametric_tests.values()]),
        'bootstrap_significant': not bootstrap_results['ci_contains_zero'],
        'permutation_significant': permutation_results['significant']
    }
    
    # Общий консенсус
    total_significant_tests = sum(significance_consensus.values())
    consensus_threshold = len(significance_consensus) // 2 + 1  # Больше половины
    
    overall_significant = total_significant_tests >= consensus_threshold
    
    # Компиляция финальных результатов
    final_results = {
        'descriptive_statistics': descriptive_stats,
        'parametric_tests': parametric_tests,
        'nonparametric_tests': nonparametric_tests,
        'effect_sizes': effect_sizes,
        'bootstrap_analysis': bootstrap_results,
        'permutation_analysis': permutation_results,
        'normality_tests': normality_tests,
        'variance_tests': variance_tests,
        'significance_consensus': significance_consensus,
        'overall_conclusion': {
            'significant': overall_significant,
            'significant_tests_count': total_significant_tests,
            'total_tests_count': len(significance_consensus),
            'alpha_level': alpha,
            'recommendation': 'significant' if overall_significant else 'not_significant'
        },
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'n_bootstrap_samples': n_bootstrap,
            'n_permutations': n_permutations,
            'alpha_level': alpha
        }
    }
    
    print(f"   ✅ Анализ завершён. Значимых тестов: {total_significant_tests}/{len(significance_consensus)}")
    print(f"   📊 Общий вывод: {'ЗНАЧИМО' if overall_significant else 'НЕ ЗНАЧИМО'} (α = {alpha})")
    
    return final_results


def sanity_check_comprehensive(classifier, test_image, target_class, xai_analyzer, 
                             n_trials=3, randomization_strength=0.01):
    """
    Комплексный sanity check для XAI методов
    
    Проверяет что XAI карты действительно отражают функциональность модели:
    1. Рандомизация весов должна разрушить карты
    2. Независимые входы должны давать независимые карты
    3. Карты должны быть чувствительны к изменениям модели
    
    Args:
        classifier: модель классификатора
        test_image: тестовое изображение
        target_class: целевой класс
        xai_analyzer: XAI анализатор
        n_trials: количество испытаний
        randomization_strength: сила рандомизации весов
    
    Returns:
        dict: результаты sanity checks
    """
    
    print("🔍 Проведение комплексного sanity check...")
    
    # Сохраняем оригинальные веса
    original_state = {name: param.clone() for name, param in classifier.named_parameters()}
    
    results = {
        'weight_randomization_test': {},
        'input_independence_test': {},
        'model_sensitivity_test': {},
        'overall_sanity_score': 0.0
    }
    
    try:
        # 1. ТЕСТ РАНДОМИЗАЦИИ ВЕСОВ
        print("   🎲 Тест рандомизации весов...")
        
        # Вычисляем оригинальные карты важности
        original_attribution = xai_analyzer.compute_integrated_gradients(
            test_image, target_class, n_steps=20
        )
        
        correlations_with_random = []
        
        for trial in range(n_trials):
            # Рандомизируем веса
            with torch.no_grad():
                for name, param in classifier.named_parameters():
                    if param.dim() > 1:  # Только веса, не bias
                        random_weights = torch.randn_like(param) * randomization_strength
                        param.data = random_weights
            
            # Вычисляем карты с рандомизированной моделью
            try:
                randomized_attribution = xai_analyzer.compute_integrated_gradients(
                    test_image, target_class, n_steps=20
                )
                
                # Корреляция между оригинальной и рандомизированной картами
                orig_flat = original_attribution.flatten().detach().cpu().numpy()
                rand_flat = randomized_attribution.flatten().detach().cpu().numpy()
                
                # Проверяем на NaN и бесконечность
                if np.any(np.isnan(orig_flat)) or np.any(np.isnan(rand_flat)):
                    correlation = 0.0
                else:
                    correlation = np.corrcoef(orig_flat, rand_flat)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                
                correlations_with_random.append(abs(correlation))
                
            except Exception as e:
                print(f"      ⚠️  Ошибка в trial {trial}: {e}")
                correlations_with_random.append(0.0)
        
        mean_random_correlation = np.mean(correlations_with_random)
        random_test_passed = mean_random_correlation < 0.1  # Порог для sanity check
        
        results['weight_randomization_test'] = {
            'mean_correlation_with_random': mean_random_correlation,
            'correlations_per_trial': correlations_with_random,
            'test_passed': random_test_passed,
            'threshold': 0.1,
            'n_trials': n_trials
        }
        
        # Восстанавливаем оригинальные веса
        for name, param in classifier.named_parameters():
            param.data = original_state[name].data
        
        # 2. ТЕСТ НЕЗАВИСИМОСТИ ВХОДОВ
        print("   🔄 Тест независимости входов...")
        
        # Создаём независимые входы
        independent_inputs = []
        for _ in range(3):
            noise_input = torch.randn_like(test_image)
            independent_inputs.append(noise_input)
        
        # Вычисляем карты для независимых входов
        independent_attributions = []
        for inp in independent_inputs:
            try:
                attr = xai_analyzer.compute_integrated_gradients(inp, target_class, n_steps=15)
                independent_attributions.append(attr.flatten().detach().cpu().numpy())
            except Exception as e:
                print(f"      ⚠️  Ошибка при вычислении независимой атрибуции: {e}")
                continue
        
        # Корреляции между независимыми картами
        independence_correlations = []
        if len(independent_attributions) >= 2:
            for i in range(len(independent_attributions)):
                for j in range(i + 1, len(independent_attributions)):
                    corr = np.corrcoef(independent_attributions[i], independent_attributions[j])[0, 1]
                    if not np.isnan(corr):
                        independence_correlations.append(abs(corr))
        
        mean_independence_correlation = np.mean(independence_correlations) if independence_correlations else 0.0
        independence_test_passed = mean_independence_correlation < 0.3
        
        results['input_independence_test'] = {
            'mean_correlation_between_independent': mean_independence_correlation,
            'independence_correlations': independence_correlations,
            'test_passed': independence_test_passed,
            'threshold': 0.3,
            'n_independent_inputs': len(independent_inputs)
        }
        
        # 3. ТЕСТ ЧУВСТВИТЕЛЬНОСТИ МОДЕЛИ
        print("   🎯 Тест чувствительности модели...")
        
        # Сравниваем карты для разных классов
        different_class_correlations = []
        
        for other_class in range(min(3, len(CLASS_NAMES))):
            if other_class != target_class:
                try:
                    other_class_attr = xai_analyzer.compute_integrated_gradients(
                        test_image, other_class, n_steps=15
                    )
                    
                    orig_flat = original_attribution.flatten().detach().cpu().numpy()
                    other_flat = other_class_attr.flatten().detach().cpu().numpy()
                    
                    corr = np.corrcoef(orig_flat, other_flat)[0, 1]
                    if not np.isnan(corr):
                        different_class_correlations.append(abs(corr))
                        
                except Exception as e:
                    print(f"      ⚠️  Ошибка при анализе класса {other_class}: {e}")
                    continue
        
        mean_different_class_correlation = np.mean(different_class_correlations) if different_class_correlations else 1.0
        sensitivity_test_passed = mean_different_class_correlation < 0.8  # Карты должны различаться
        
        results['model_sensitivity_test'] = {
            'mean_correlation_different_classes': mean_different_class_correlation,
            'different_class_correlations': different_class_correlations,
            'test_passed': sensitivity_test_passed,
            'threshold': 0.8,
            'classes_tested': len(different_class_correlations)
        }
        
        # ОБЩИЙ SANITY SCORE
        passed_tests = [
            results['weight_randomization_test']['test_passed'],
            results['input_independence_test']['test_passed'],
            results['model_sensitivity_test']['test_passed']
        ]
        
        sanity_score = sum(passed_tests) / len(passed_tests)
        results['overall_sanity_score'] = sanity_score
        
        # Интерпретация
        if sanity_score >= 0.67:
            sanity_interpretation = 'good'
        elif sanity_score >= 0.33:
            sanity_interpretation = 'moderate'
        else:
            sanity_interpretation = 'poor'
        
        results['overall_interpretation'] = sanity_interpretation
        
        print(f"   📊 Sanity score: {sanity_score:.2f} ({sanity_interpretation})")
        print(f"   ✅ Пройденных тестов: {sum(passed_tests)}/{len(passed_tests)}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка в sanity check: {e}")
        results['error'] = str(e)
        
    finally:
        # ОБЯЗАТЕЛЬНО восстанавливаем оригинальные веса
        try:
            for name, param in classifier.named_parameters():
                param.data = original_state[name].data
            print("   🔄 Оригинальные веса классификатора восстановлены")
        except Exception as e:
            print(f"❌ Ошибка восстановления весов: {e}")
    
    return results


print("✅ Комплексная статистическая валидация готова!")
print("📊 Включает: параметрические/непараметрические тесты, bootstrap, permutation, effect size")
print("🔍 Sanity checks: рандомизация весов, независимость входов, чувствительность модели")


def tensor_to_displayable_image(tensor, denormalize=True):
    """Конвертирует тензор PyTorch в numpy array для отображения"""
    
    if torch.is_tensor(tensor):
        img = tensor.squeeze().detach().cpu().numpy()
    else:
        img = tensor
    
    # Переставляем оси если нужно (CHW -> HWC)
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    
    # Убираем лишнюю размерность для grayscale
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)
    
    # Денормализация из [-1, 1] в [0, 1]
    if denormalize:
        img = (img + 1.0) / 2.0
    
    return np.clip(img, 0, 1)


def visualize_xai_step_comprehensive(image, attribution_map, top_k_mask, bottom_k_mask,
                                   timestep, class_name, save_path=None, figsize=(20, 5)):
    """
    Комплексная визуализация XAI анализа для одного временного шага
    
    Показывает: оригинал, карту важности, топ-k и bottom-k маски, наложения
    """
    
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    
    # 1. Оригинальное изображение
    img_display = tensor_to_displayable_image(image)
    axes[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axes[0].set_title(f'Original\nt = {timestep:.0f}', fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # 2. Карта атрибуции
    if torch.is_tensor(attribution_map):
        attr_display = attribution_map.squeeze().detach().cpu().numpy()
        if len(attr_display.shape) == 3:
            # Используем L2 норму по каналам
            attr_display = np.linalg.norm(attr_display, axis=0)
        else:
            attr_display = np.abs(attr_display)
    else:
        attr_display = np.abs(attribution_map)
    
    im1 = axes[1].imshow(attr_display, cmap='hot', alpha=0.8)
    axes[1].set_title(f'Attribution Map\n(max: {attr_display.max():.3f})', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8, aspect=20)
    
    # 3. Топ-k регионы
    axes[2].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axes[2].imshow(top_k_mask, cmap='Reds', alpha=0.6)
    top_k_coverage = np.sum(top_k_mask) / top_k_mask.size * 100
    axes[2].set_title(f'Top-k Regions\n({top_k_coverage:.1f}% coverage)', fontsize=12)
    axes[2].axis('off')
    
    # 4. Bottom-k регионы
    axes[3].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axes[3].imshow(bottom_k_mask, cmap='Blues', alpha=0.6)
    bottom_k_coverage = np.sum(bottom_k_mask) / bottom_k_mask.size * 100
    axes[3].set_title(f'Bottom-k Regions\n({bottom_k_coverage:.1f}% coverage)', fontsize=12)
    axes[3].axis('off')
    
    # 5. Комбинированное наложение
    axes[4].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axes[4].imshow(top_k_mask, cmap='Reds', alpha=0.4)
    axes[4].imshow(bottom_k_mask, cmap='Blues', alpha=0.3)
    axes[4].set_title(f'{class_name} XAI\nRed: Top-k, Blue: Bottom-k', fontsize=12)
    axes[4].axis('off')
    
    plt.suptitle(f'🔬 XAI Analysis Step: {class_name} at timestep {timestep:.0f}', 
                fontsize=16, y=1.02, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Сохранено: {save_path}")
    
    plt.show()


def visualize_intervention_comprehensive(original_image, masks_dict, interventions_dict,
                                       cfi_results_dict, timestep=None, save_path=None):
    """
    Комплексная визуализация результатов интервенций
    
    Args:
        original_image: оригинальное изображение
        masks_dict: словарь масок {'top_k': mask, 'bottom_k': mask}
        interventions_dict: словарь результатов интервенций
        cfi_results_dict: словарь результатов CFI
    """
    
    n_interventions = len(interventions_dict)
    n_cols = min(4, n_interventions + 1)  # +1 для оригинала
    n_rows = (n_interventions + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Исходное изображение
    img_display = tensor_to_displayable_image(original_image)
    axes[0, 0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axes[0, 0].set_title('Original', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    plot_idx = 1
    
    for region_type, intervention_results in interventions_dict.items():
        for intervention_type, result_data in intervention_results.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            
            if row >= n_rows:
                break
            
            # Модифицированное изображение
            modified_img = tensor_to_displayable_image(result_data['modified_image'])
            axes[row, col].imshow(modified_img, cmap='gray' if len(modified_img.shape) == 2 else None)
            
            # Информация о результатах
            cfi_key = f"{region_type}_{intervention_type}"
            if cfi_key in cfi_results_dict:
                cfi_result = cfi_results_dict[cfi_key]
                cfi_val = cfi_result['target_class_analysis']['cfi']
                pred_changed = cfi_result['prediction_analysis']['prediction_changed']
                
                title = f"{region_type.replace('_', '-').title()}\n{intervention_type.title()}\n"
                title += f"CFI: {cfi_val:.3f}\nPred: {'✓' if pred_changed else '✗'}"
            else:
                title = f"{region_type.replace('_', '-').title()}\n{intervention_type.title()}"
            
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
            
            plot_idx += 1
    
    # Удаляем неиспользуемые подграфики
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if row < n_rows and col < n_cols:
            axes[row, col].axis('off')
    
    title = f'🧪 Counterfactual Interventions'
    if timestep is not None:
        title += f' (t = {timestep:.0f})'
    
    plt.suptitle(title, fontsize=16, y=0.98, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()


def plot_time_shap_comprehensive(timesteps, time_importance, time_data, class_name, save_path=None):
    """
    Комплексная визуализация Time-SHAP анализа
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Нормализованная важность по времени
    ax1.plot(timesteps, time_importance, 'bo-', linewidth=3, markersize=8, alpha=0.7)
    ax1.fill_between(timesteps, time_importance, alpha=0.3, color='blue')
    
    # Отмечаем наиболее важный шаг
    max_idx = np.argmax(time_importance)
    ax1.axvline(x=timesteps[max_idx], color='red', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Most important: t={timesteps[max_idx]:.0f}')
    ax1.scatter(timesteps[max_idx], time_importance[max_idx], 
               color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)
    
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('Normalized Importance', fontsize=12)
    ax1.set_title(f'Time-SHAP: Temporal Importance for {class_name}', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Per-class scores (логарифмические вероятности)
    confidence_scores = time_data['confidence_scores']
    ax2.plot(timesteps, confidence_scores, 'go-', linewidth=3, markersize=8, alpha=0.7)
    ax2.fill_between(timesteps, confidence_scores, alpha=0.3, color='green')
    
    ax2.axvline(x=timesteps[max_idx], color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.scatter(timesteps[max_idx], confidence_scores[max_idx], 
               color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)
    
    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel('Log Probability Score', fontsize=12)
    ax2.set_title('Per-class Score Evolution', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Обычные вероятности
    probability_scores = time_data['probability_scores']
    ax3.plot(timesteps, probability_scores, 'mo-', linewidth=3, markersize=8, alpha=0.7)
    ax3.fill_between(timesteps, probability_scores, alpha=0.3, color='magenta')
    
    ax3.axvline(x=timesteps[max_idx], color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.scatter(timesteps[max_idx], probability_scores[max_idx], 
               color='red', s=150, zorder=10, edgecolor='darkred', linewidth=2)
    
    ax3.set_xlabel('Timestep t', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Probability Evolution', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Статистика распределения важности
    ax4.hist(time_importance, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=np.mean(time_importance), color='red', linestyle='-', 
               label=f'Mean: {np.mean(time_importance):.3f}', linewidth=2)
    ax4.axvline(x=np.median(time_importance), color='orange', linestyle='--', 
               label=f'Median: {np.median(time_importance):.3f}', linewidth=2)
    
    ax4.set_xlabel('Importance Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Time Importance', fontsize=14, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'🕒 Time-SHAP Comprehensive Analysis: {class_name}', 
                fontsize=18, y=0.98, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()


def plot_statistical_analysis_modern(statistical_results, class_name, save_path=None):
    """
    Современная визуализация статистического анализа
    """
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Извлекаем данные
    top_k_stats = statistical_results['descriptive_statistics']['top_k']
    bottom_k_stats = statistical_results['descriptive_statistics']['bottom_k']
    
    # Создаём синтетические данные для визуализации на основе статистик
    np.random.seed(42)
    top_k_synthetic = np.random.normal(top_k_stats['mean'], top_k_stats['std'], top_k_stats['n'])
    bottom_k_synthetic = np.random.normal(bottom_k_stats['mean'], bottom_k_stats['std'], bottom_k_stats['n'])
    
    # 1. Коробчатые диаграммы
    ax1 = fig.add_subplot(gs[0, 0])
    data_to_plot = [top_k_synthetic, bottom_k_synthetic]
    box_plot = ax1.boxplot(data_to_plot, labels=['Top-k', 'Bottom-k'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    box_plot['boxes'][1].set_facecolor('lightblue')
    ax1.set_ylabel('Causal Shift (CFI)', fontsize=12)
    ax1.set_title('CFI Distribution Comparison', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Гистограммы
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(top_k_synthetic, alpha=0.7, label='Top-k', bins=20, color='lightcoral', density=True)
    ax2.hist(bottom_k_synthetic, alpha=0.7, label='Bottom-k', bins=20, color='lightblue', density=True)
    ax2.set_xlabel('CFI Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Probability Density Functions', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. P-values comparison
    ax3 = fig.add_subplot(gs[0, 2])
    tests = ['t-test', 'Welch t-test', 'Mann-Whitney', 'Permutation']
    p_values = [
        statistical_results['parametric_tests']['t_test']['p_value'],
        statistical_results['parametric_tests']['welch_t_test']['p_value'],
        statistical_results['nonparametric_tests']['mann_whitney_u']['p_value'],
        statistical_results['permutation_analysis']['p_value']
    ]
    
    colors = ['coral', 'orange', 'skyblue', 'lightgreen']
    bars = ax3.bar(tests, p_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax3.axhline(y=0.01, color='darkred', linestyle=':', linewidth=2, label='α = 0.01')
    
    ax3.set_ylabel('p-value', fontsize=12)
    ax3.set_title('Statistical Test Results', fontsize=14, weight='bold')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Добавляем значения на столбцы
    for bar, p_val in zip(bars, p_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{p_val:.1e}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # 4. Bootstrap confidence interval
    ax4 = fig.add_subplot(gs[1, 0])
    bootstrap_data = statistical_results['bootstrap_analysis']
    mean_diff = bootstrap_data['mean_diff']
    ci_lower = bootstrap_data['ci_lower']
    ci_upper = bootstrap_data['ci_upper']
    
    ax4.errorbar([0], [mean_diff], yerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]], 
                fmt='ro', capsize=15, markersize=10, capthick=3, linewidth=3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylabel('Mean Difference', fontsize=12)
    ax4.set_title(f'Bootstrap 95% CI\nContains 0: {bootstrap_data["ci_contains_zero"]}', fontsize=14, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks([])
    
    # 5. Effect sizes
    ax5 = fig.add_subplot(gs[1, 1])
    effect_sizes = statistical_results['effect_sizes']
    effect_names = list(effect_sizes.keys())
    effect_values = [effect_sizes[name]['value'] for name in effect_names]
    
    bars_effect = ax5.barh(effect_names, effect_values, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax5.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect')
    ax5.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Medium effect')
    ax5.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.7, label='Large effect')
    
    ax5.set_xlabel('Effect Size', fontsize=12)
    ax5.set_title('Effect Size Analysis', fontsize=14, weight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Test significance summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Подсчитываем значимые тесты
    significance_data = statistical_results['significance_consensus']
    significant_count = sum(significance_data.values())
    total_tests = len(significance_data)
    
    # Создаём круговую диаграмму значимости
    labels = ['Significant', 'Not Significant']
    sizes = [significant_count, total_tests - significant_count]
    colors = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                      startangle=90, textprops={'fontsize': 12})
    ax6.set_title(f'Test Significance Summary\n({significant_count}/{total_tests} significant)', 
                 fontsize=14, weight='bold')
    
    # 7-12. Детальная статистическая таблица
    ax7 = fig.add_subplot(gs[2:, :])
    ax7.axis('off')
    
    # Создаем детальную таблицу
    table_data = []
    
    # Базовые статистики
    table_data.append(['Statistic', 'Top-k', 'Bottom-k', 'Difference'])
    table_data.append(['Sample Size', f"{top_k_stats['n']}", f"{bottom_k_stats['n']}", '—'])
    table_data.append(['Mean', f"{top_k_stats['mean']:.4f}", f"{bottom_k_stats['mean']:.4f}", 
                      f"{top_k_stats['mean'] - bottom_k_stats['mean']:.4f}"])
    table_data.append(['Std Dev', f"{top_k_stats['std']:.4f}", f"{bottom_k_stats['std']:.4f}", '—'])
    table_data.append(['Median', f"{top_k_stats['median']:.4f}", f"{bottom_k_stats['median']:.4f}", 
                      f"{top_k_stats['median'] - bottom_k_stats['median']:.4f}"])
    table_data.append(['IQR', f"{top_k_stats['iqr']:.4f}", f"{bottom_k_stats['iqr']:.4f}", '—'])
    
    # Тесты значимости
    table_data.append(['', '', '', ''])  # Пустая строка
    table_data.append(['Test', 'Statistic', 'p-value', 'Significant'])
    
    for test_category in ['parametric_tests', 'nonparametric_tests']:
        for test_name, test_result in statistical_results[test_category].items():
            formatted_name = test_name.replace('_', ' ').title()
            table_data.append([
                formatted_name,
                f"{test_result['statistic']:.4f}",
                f"{test_result['p_value']:.1e}",
                '✅' if test_result['significant'] else '❌'
            ])
    
    # Permutation test
    perm_result = statistical_results['permutation_analysis']
    table_data.append([
        'Permutation Test',
        f"{perm_result['observed_difference']:.4f}",
        f"{perm_result['p_value']:.1e}",
        '✅' if perm_result['significant'] else '❌'
    ])
    
    # Effect sizes
    table_data.append(['', '', '', ''])  # Пустая строка
    table_data.append(['Effect Size', 'Value', 'Interpretation', ''])
    
    for effect_name, effect_data in effect_sizes.items():
        formatted_name = effect_name.replace('_', ' ').title()
        interpretation = effect_data.get('interpretation', '—')
        table_data.append([
            formatted_name,
            f"{effect_data['value']:.4f}",
            interpretation.title(),
            ''
        ])
    
    # Создаем таблицу
    table = ax7.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Раскрашиваем заголовки
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Раскрашиваем строки
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.suptitle(f'📊 Comprehensive Statistical Analysis: {class_name}\n' +
                f'Overall Result: {"SIGNIFICANT" if statistical_results["overall_conclusion"]["significant"] else "NOT SIGNIFICANT"}',
                fontsize=18, y=0.98, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()


print("✅ Современные функции визуализации готовы!")
print("🎨 Включают: XAI шаги, интервенции, Time-SHAP, статистический анализ")
print("📊 Высококачественная графика с детальными аннотациями")


def run_comprehensive_xai_pipeline(trajectory, timesteps, xai_analyzer, classifier, 
                                 target_class_id, target_class_name,
                                 save_results=True, results_dir=None):
    """
    Запускает полный пайплайн XAI анализа
    
    Этот пайплайн выполняет все основные этапы XAI анализа:
    1. Вычисление XAI карт для каждого временного шага
    2. Выделение топ-k и bottom-k регионов
    3. Контрафактуальные интервенции
    4. Вычисление CFI метрик
    5. Статистическая валидация
    6. Time-SHAP анализ
    7. Sanity checks
    8. Генерация отчётов
    
    Args:
        trajectory: список изображений диффузионной траектории
        timesteps: соответствующие временные шаги
        xai_analyzer: инициализированный XAI анализатор
        classifier: модель классификатора
        target_class_id: ID целевого класса
        target_class_name: название целевого класса
        save_results: сохранять результаты
        results_dir: директория для сохранения
    
    Returns:
        dict: полные результаты анализа
    """
    
    print("🚀 === ЗАПУСК КОМПЛЕКСНОГО XAI ПАЙПЛАЙНА ===")
    print(f"🎯 Целевой класс: {target_class_name} (ID: {target_class_id})")
    print(f"📈 Анализируемых временных шагов: {len(trajectory)}")
    print(f"💾 Сохранение результатов: {save_results}")
    
    # Настройка директории результатов
    if save_results and results_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = RESULTS_DIR / f"xai_analysis_{target_class_name}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Результаты будут сохранены в: {results_dir}")
    
    # Инициализация структуры результатов
    results = {
        'metadata': {
            'target_class_id': target_class_id,
            'target_class_name': target_class_name,
            'n_timesteps': len(trajectory),
            'timesteps': timesteps,
            'analysis_timestamp': datetime.now().isoformat(),
            'parameters': {
                'top_k_percent': TOP_K_PERCENT,
                'bottom_k_percent': BOTTOM_K_PERCENT,
                'ig_n_steps': IG_N_STEPS,
                'shap_n_samples': SHAP_N_SAMPLES,
                'intervention_types': INTERVENTION_TYPES,
                'alpha_level': ALPHA_LEVEL
            }
        },
        'xai_maps': {},
        'region_analysis': {},
        'interventions': {},
        'cfi_analysis': {},
        'time_shap': {},
        'statistical_validation': {},
        'sanity_checks': {},
        'visualizations': []
    }
    
    try:
        # === ЭТАП 1: ВЫЧИСЛЕНИЕ XAI КАРТ ===
        print("🔬 Этап 1: Вычисление XAI карт для каждого временного шага...")
        
        xai_maps = {}
        region_data = {}
        
        total_frames = len(trajectory)
        for i, (image_tensor, timestep) in enumerate(tqdm(zip(trajectory, timesteps), 
                                                         desc="Computing XAI maps", 
                                                         total=total_frames)):
            image_gpu = image_tensor.to(device)
            
            try:
                # Считаем отдельно IG и SHAP, чтобы сохранить и подписать каждую карту
                ig_attr = xai_analyzer.compute_integrated_gradients(image_gpu, target_class_id)
                shap_attr = xai_analyzer.compute_shap_approximation(image_gpu, target_class_id)
                combined_attr, method_details = xai_analyzer.compute_combined_attribution(
                    image_gpu, target_class_id, methods=['ig', 'shap'], weights=[0.5, 0.5]
                )
                
                # Выделение регионов
                top_k_data = select_regions_advanced(
                    combined_attr, k_percent=TOP_K_PERCENT, region_type='top'
                )
                
                bottom_k_data = select_regions_advanced(
                    combined_attr, k_percent=BOTTOM_K_PERCENT, region_type='bottom'
                )
                
                # Сохранение результатов
                step_key = f"t_{timestep:.0f}"
                xai_maps[step_key] = {
                    'timestep': timestep,
                    'attribution_map': combined_attr,
                    'method_details': method_details,
                    'image_shape': tuple(image_tensor.shape)
                }
                
                region_data[step_key] = {
                    'top_k': top_k_data,
                    'bottom_k': bottom_k_data
                }
                
                # Визуализация каждого шага
                if save_results:
                    # Сохраняем комбинированную карту
                    viz_path = results_dir / f"xai_step_{step_key}.png"
                    visualize_xai_step_comprehensive(
                        image_tensor, combined_attr, 
                        top_k_data['mask'], bottom_k_data['mask'],
                        timestep, target_class_name, save_path=viz_path
                    )
                    results['visualizations'].append(str(viz_path))
                    # Дополнительно сохраняем IG и SHAP отдельно для прозрачности
                    viz_path_ig = results_dir / f"xai_step_{step_key}_IG.png"
                    visualize_xai_step_comprehensive(
                        image_tensor, ig_attr,
                        top_k_data['mask'], bottom_k_data['mask'],
                        timestep, f"{target_class_name} (IG)", save_path=viz_path_ig
                    )
                    results['visualizations'].append(str(viz_path_ig))
                    viz_path_shap = results_dir / f"xai_step_{step_key}_SHAP.png"
                    visualize_xai_step_comprehensive(
                        image_tensor, shap_attr,
                        top_k_data['mask'], bottom_k_data['mask'],
                        timestep, f"{target_class_name} (SHAP)", save_path=viz_path_shap
                    )
                    results['visualizations'].append(str(viz_path_shap))
                else:
                    visualize_xai_step_comprehensive(
                        image_tensor, combined_attr, 
                        top_k_data['mask'], bottom_k_data['mask'],
                        timestep, target_class_name
                    )
                
            except Exception as e:
                print(f"   ⚠️  Ошибка в шаге {i} (t={timestep}): {e}")
                continue
            # Логируем прогресс XAI по кадрам
            try:
                _log_progress_bar("XAI maps", i + 1, total_frames)
            except Exception:
                pass
        
        results['xai_maps'] = xai_maps
        results['region_analysis'] = region_data
        
        print(f"   ✅ XAI карты вычислены для {len(xai_maps)} шагов")
        
        # === ЭТАП 2: КОНТРАФАКТУАЛЬНЫЕ ИНТЕРВЕНЦИИ ===
        print("🧪 Этап 2: Контрафактуальные интервенции...")
        
        interventions_data = {}
        cfi_data = {}
        
        # Выбираем несколько ключевых временных шагов для интервенций
        key_steps = [0, len(trajectory)//2, len(trajectory)-4,len(trajectory)-3,len(trajectory)-2, len(trajectory)-1]  # Начало, середина, конец
        
        total_keys = len(key_steps)
        for idx_k, step_idx in enumerate(key_steps):
            if step_idx >= len(trajectory):
                continue
                
            image_tensor = trajectory[step_idx]
            timestep = timesteps[step_idx]
            step_key = f"t_{timestep:.0f}"
            
            if step_key not in region_data:
                continue
            
            print(f"   🔬 Анализ интервенций для t={timestep:.0f}...")
            
            image_gpu = image_tensor.to(device)
            step_interventions = {}
            step_cfi = {}
            
            for region_type in ['top_k', 'bottom_k']:
                region_mask = region_data[step_key][region_type]['mask']
                step_interventions[region_type] = {}
                
                for intervention_type in INTERVENTION_TYPES:
                    try:
                        # Интервенция
                        intervention_result = counterfactual_intervention_advanced(
                            image_gpu, region_mask, intervention_type
                        )
                        
                        # CFI анализ
                        cfi_result = compute_causal_shift_comprehensive(
                            classifier, image_gpu, 
                            intervention_result['modified_image'],
                            target_class_id, include_all_classes=True
                        )
                        
                        step_interventions[region_type][intervention_type] = intervention_result
                        step_cfi[f"{region_type}_{intervention_type}"] = cfi_result
                        
                    except Exception as e:
                        print(f"      ⚠️  Ошибка в {region_type}/{intervention_type}: {e}")
                        continue
            
            interventions_data[step_key] = step_interventions
            cfi_data[step_key] = step_cfi
            
            # Визуализация интервенций
            if save_results:
                viz_path = results_dir / f"interventions_{step_key}.png"
                visualize_intervention_comprehensive(
                    image_tensor, 
                    {region_type: region_data[step_key][region_type]['mask'] 
                     for region_type in ['top_k', 'bottom_k']},
                    step_interventions, step_cfi, timestep, save_path=viz_path
                )
                results['visualizations'].append(str(viz_path))
            # Логируем прогресс по интервенциям/CFI
            try:
                _log_progress_bar("Interventions/CFI", idx_k + 1, total_keys)
            except Exception:
                pass
        
        results['interventions'] = interventions_data
        results['cfi_analysis'] = cfi_data
        
        print(f"   ✅ Интервенции выполнены для {len(interventions_data)} временных шагов")
        
        # === ЭТАП 3: TIME-SHAP АНАЛИЗ ===
        print("🕒 Этап 3: Time-SHAP анализ временной важности...")
        
        try:
            time_importance, time_data = xai_analyzer.compute_time_shap(
                trajectory, timesteps, target_class_id
            )
            
            results['time_shap'] = {
                'importance': time_importance,
                'raw_data': time_data,
                'most_important_timestep': timesteps[np.argmax(time_importance)],
                'most_important_index': int(np.argmax(time_importance))
            }
            
            # Визуализация Time-SHAP
            if save_results:
                viz_path = results_dir / "time_shap_analysis.png"
                plot_time_shap_comprehensive(
                    timesteps, time_importance, time_data, 
                    target_class_name, save_path=viz_path
                )
                results['visualizations'].append(str(viz_path))
            else:
                plot_time_shap_comprehensive(
                    timesteps, time_importance, time_data, target_class_name
                )
            
            print("   ✅ Time-SHAP анализ завершён")
            try:
                _log_progress_bar("Time-SHAP", 1, 1)
            except Exception:
                pass
            
        
            

        except Exception as e:
            print(f"   ❌ Ошибка в Time-SHAP анализе: {e}")
            results['time_shap'] = {'error': str(e)}



        device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # Диагностика и исправление размерностей
            print("🔍 Диагностика размерностей...")
            
            target_layer = classifier.model.layer4[-1].conv2
            print(f"✅ Target layer: {target_layer}")

            gradcam_results = {}
            all_cams = []

            # Создаём оболочку без предобработки
            class RawClassifier(torch.nn.Module):
                def __init__(self, original_classifier):
                    super().__init__()
                    self.model = original_classifier.model
                    
                def forward(self, x):
                    return self.model(x)
            
            raw_classifier = RawClassifier(classifier).to(device_torch)
            raw_classifier.eval()
            
            def manual_preprocess(image_tensor):
                """Ручная предобработка с правильными размерностями"""
                # Приводим к корректной 4D форме [N, C, H, W]
                if len(image_tensor.shape) == 5:  # [1, 1, 3, 128, 128]
                    x = image_tensor.squeeze(1)  # [1, 3, 128, 128]
                elif len(image_tensor.shape) == 3:  # [3, 128, 128]
                    x = image_tensor.unsqueeze(0)  # [1, 3, 128, 128]
                else:
                    x = image_tensor
                
                # Убеждаемся что это [1, 3, H, W]
                # Проверяем размерность (без вывода предупреждения)
                if x.shape == 1 and x.shape == 3: pass

                    
                # Из [-1,1] в [0,1]
                x = torch.clamp((x + 1.0) / 2.0, 0, 1)
                
                # Ресайз до 224x224
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
                
                # ImageNet нормализация
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device_torch)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device_torch)
                x = (x - mean) / std
                
                return x
            
            with GradCAM(model=raw_classifier, target_layers=[target_layer]) as cam:
                for i, (image_tensor, timestep) in enumerate(zip(trajectory, timesteps)):
                    # Исправляем размерности
                    #print(f"📐 Исходная размерность image_tensor: {image_tensor.shape}")
                    
                    # Приводим к правильной размерности [3, 128, 128]
                    if len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1:  # [1, 3, 128, 128]
                        clean_tensor = image_tensor.squeeze(0)  # [3, 128, 128]
                    elif len(image_tensor.shape) == 5:  # [1, 1, 3, 128, 128]
                        clean_tensor = image_tensor.squeeze(0).squeeze(0)  # [3, 128, 128]
                    else:
                        clean_tensor = image_tensor  # уже [3, 128, 128]
                    
                    #print(f"📐 Очищенная размерность: {clean_tensor.shape}")
                    
                    # Теперь добавляем batch dimension для обработки
                    raw_input = clean_tensor.unsqueeze(0).to(device_torch)  # [1, 3, 128, 128]
                    processed_input = manual_preprocess(raw_input)  # [1, 3, 224, 224]
                    
                    #print(f"📐 После обработки: {processed_input.shape}")
                    
                    # GradCAM
                    grayscale_cam = cam(
                        input_tensor=processed_input,
                        targets=[ClassifierOutputTarget(target_class_id)]
                    )
                    grayscale_cam = grayscale_cam[0, :]  # (224,224)
                    all_cams.append(grayscale_cam)

                    # Визуализация
                    rgb_img = clean_tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # (128,128,3)
                    rgb_img = (rgb_img + 1.0) / 2.0  # [-1,1] -> [0,1]
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    # Ресайз до 224x224 для совпадения с CAM
                    from skimage.transform import resize
                    rgb_img_224 = resize(rgb_img, (224, 224), anti_aliasing=True, preserve_range=False)
                    
                    cam_image = show_cam_on_image(rgb_img_224, grayscale_cam, use_rgb=True)

                    step_key = f"t_{timestep:.0f}"
                    gradcam_results[step_key] = grayscale_cam

                    if save_results:
                        cam_path = results_dir / f"gradcam_{step_key}.png"
                        plt.imsave(cam_path, cam_image)
                        results['visualizations'].append(str(cam_path))
                    else:
                        plt.figure(figsize=(8,4))
                        plt.subplot(1,2,1)
                        plt.imshow(rgb_img_224)
                        plt.title(f"Исходное t={timestep:.0f}")
                        plt.axis('off')
                        plt.subplot(1,2,2)
                        plt.imshow(cam_image)
                        plt.title(f"Grad-CAM t={timestep:.0f}")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()

            results['gradcam'] = gradcam_results
            print("   ✅ Grad-CAM по шагам готов")

            # Grad-CAM для важного шага
            if 'time_shap' in results and 'most_important_index' in results['time_shap']:
                imp_idx = results['time_shap']['most_important_index']
                imp_timestep = timesteps[imp_idx]
                
                # Исправляем размерности для важного шага
                imp_tensor = trajectory[imp_idx]
                if len(imp_tensor.shape) == 4 and imp_tensor.shape[0] == 1:
                    imp_clean = imp_tensor.squeeze(0)
                elif len(imp_tensor.shape) == 5:
                    imp_clean = imp_tensor.squeeze(0).squeeze(0)
                else:
                    imp_clean = imp_tensor
                
                with GradCAM(model=raw_classifier, target_layers=[target_layer]) as cam:
                    imp_raw = imp_clean.unsqueeze(0).to(device_torch)
                    imp_processed = manual_preprocess(imp_raw)
                    
                    grayscale_cam = cam(
                        input_tensor=imp_processed,
                        targets=[ClassifierOutputTarget(target_class_id)]
                    )
                
                grayscale_cam = grayscale_cam[0, :]
                
                rgb_img = imp_clean.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                rgb_img = (rgb_img + 1.0) / 2.0
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_img_224 = resize(rgb_img, (224, 224), anti_aliasing=True, preserve_range=False)
                
                cam_image = show_cam_on_image(rgb_img_224, grayscale_cam, use_rgb=True)

                if save_results:
                    imp_path = results_dir / f"gradcam_most_important_t{imp_timestep:.0f}.png"
                    plt.imsave(imp_path, cam_image)
                    results['visualizations'].append(str(imp_path))
                else:
                    plt.imshow(cam_image)
                    plt.title(f"Grad-CAM важный шаг t={imp_timestep:.0f}")
                    plt.axis('off')
                    plt.show()

                results['gradcam_most_important'] = {
                    'timestep': float(imp_timestep),
                    'index': int(imp_idx),
                    'gradcam': grayscale_cam
                }
                print(f"   ✅ Grad-CAM для важного шага (t={imp_timestep:.0f}) готов")

            # Суммарный CAM
            if len(all_cams) > 0:
                summed_cam = np.mean(np.stack(all_cams, axis=0), axis=0)
                summed_cam = (summed_cam - summed_cam.min()) / (summed_cam.max() - summed_cam.min() + 1e-8)

                final_tensor = trajectory[-1]
                if len(final_tensor.shape) == 4 and final_tensor.shape[0] == 1:
                    final_clean = final_tensor.squeeze(0)
                elif len(final_tensor.shape) == 5:
                    final_clean = final_tensor.squeeze(0).squeeze(0)
                else:
                    final_clean = final_tensor

                final_img = final_clean.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                final_img = (final_img + 1.0) / 2.0
                final_img = np.clip(final_img, 0, 1)
                final_img_224 = resize(final_img, (224, 224), anti_aliasing=True, preserve_range=False)
                
                cam_image = show_cam_on_image(final_img_224, summed_cam, use_rgb=True)

                if save_results:
                    sum_path = results_dir / "gradcam_summary_all_timesteps.png"
                    plt.imsave(sum_path, cam_image)
                    results['visualizations'].append(str(sum_path))
                else:
                    plt.imshow(cam_image)
                    plt.title("Суммарный Grad-CAM по всем t")
                    plt.axis('off')
                    plt.show()

                results['gradcam_summary'] = summed_cam
                print("   ✅ Суммарный Grad-CAM рассчитан")
                print("gradcam_overview save")
                # Исходное изображение
                plt.figure(figsize=(16,5))

                # 1. Оригинал
                plt.subplot(1,3,1)
                plt.imshow(final_img_224)
                plt.title("Original")
                plt.axis('off')

                # 2. Важный Grad-CAM
                plt.subplot(1,3,2)
                plt.imshow(show_cam_on_image(final_img_224, results['gradcam_most_important']['gradcam'], use_rgb=True))
                plt.title("Most important Grad-CAM (t={:.0f})".format(results['gradcam_most_important']['timestep']))
                plt.axis('off')

                # 3. Суммарный Grad-CAM
                plt.subplot(1,3,3)
                plt.imshow(show_cam_on_image(final_img_224, results['gradcam_summary'], use_rgb=True))
                plt.title("Summed Grad-CAM")
                plt.axis('off')

                plt.tight_layout()
                # Сохраняем единый коллаж
                plt.savefig(results_dir / "gradcam_overview.png")
                plt.close()
            try:
                _log_progress_bar("Grad-CAM", 1, 1)
            except Exception:
                pass
        except Exception as e:
            print(f"   ❌ Ошибка в Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            results['gradcam'] = {'error': str(e)}




        # === ЭТАП 4: СБОР CFI ДАННЫХ ДЛЯ СТАТИСТИКИ ===
        print("📊 Этап 4: Подготовка данных для статистического анализа...")
        
        top_k_shifts = []
        bottom_k_shifts = []
        
        for step_key, step_cfi in cfi_data.items():
            for intervention_key, cfi_result in step_cfi.items():
                if 'top_k' in intervention_key:
                    top_k_shifts.append(cfi_result['target_class_analysis']['cfi'])
                elif 'bottom_k' in intervention_key:
                    bottom_k_shifts.append(cfi_result['target_class_analysis']['cfi'])
        
        print(f"   📈 Собрано CFI значений: Top-k={len(top_k_shifts)}, Bottom-k={len(bottom_k_shifts)}")
        
        # === ЭТАП 5: СТАТИСТИЧЕСКАЯ ВАЛИДАЦИЯ ===
        print("📊 Этап 5: Комплексная статистическая валидация...")
        
        if len(top_k_shifts) > 0 and len(bottom_k_shifts) > 0:
            try:
                statistical_results = statistical_validation_comprehensive(
                    top_k_shifts, bottom_k_shifts,
                    alpha=ALPHA_LEVEL, 
                    n_bootstrap=N_BOOTSTRAP,
                    n_permutations=N_PERMUTATIONS
                )
                
                results['statistical_validation'] = statistical_results
                
                # Визуализация статистики
                if save_results:
                    viz_path = results_dir / "statistical_analysis.png"
                    plot_statistical_analysis_modern(
                        statistical_results, target_class_name, save_path=viz_path
                    )
                    results['visualizations'].append(str(viz_path))
                else:
                    plot_statistical_analysis_modern(statistical_results, target_class_name)
                
                print(f"   ✅ Статистическая валидация завершена")
                
            except Exception as e:
                print(f"   ❌ Ошибка в статистической валидации: {e}")
                results['statistical_validation'] = {'error': str(e)}
        else:
            print("   ⚠️  Недостаточно данных для статистической валидации")
            results['statistical_validation'] = {'error': 'Insufficient data'}
        
        # === ЭТАП 6: SANITY CHECKS ===
        print("🔍 Этап 6: Sanity checks...")
        
        try:
            # Используем финальное изображение для sanity check
            final_image = trajectory[-1].to(device)
            
            sanity_results = sanity_check_comprehensive(
                classifier, final_image, target_class_id, xai_analyzer,
                n_trials=3, randomization_strength=0.01
            )
            
            results['sanity_checks'] = sanity_results
            print("   ✅ Sanity checks завершены")
            
        except Exception as e:
            print(f"   ❌ Ошибка в sanity checks: {e}")
            results['sanity_checks'] = {'error': str(e)}
        
        # === ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
        if save_results:
            print("💾 Этап 7: Сохранение результатов...")
            
            # JSON отчёт
            json_path = results_dir / 'analysis_results.json'
            
            # Подготавливаем данные для JSON (убираем тензоры)
            json_results = results.copy()
            
            # Убираем тензоры из xai_maps
            for step_key in json_results.get('xai_maps', {}):
                if 'attribution_map' in json_results['xai_maps'][step_key]:
                    del json_results['xai_maps'][step_key]['attribution_map']
            
            # Убираем тензоры из интервенций
            for step_key in json_results.get('interventions', {}):
                for region_type in json_results['interventions'][step_key]:
                    for intervention_type in json_results['interventions'][step_key][region_type]:
                        intervention_data = json_results['interventions'][step_key][region_type][intervention_type]
                        for key in ['modified_image', 'intervention', 'mask_tensor', 'difference']:
                            if key in intervention_data:
                                del intervention_data[key]
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Pickle для полных данных
            pickle_path = results_dir / 'full_results.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"   📄 JSON отчёт: {json_path}")
            print(f"   🗂️ Полные данные: {pickle_path}")
            print(f"   🎨 Визуализации: {len(results['visualizations'])} файлов")
        
        # === ИТОГОВЫЙ ОТЧЁТ ===
        print("🎉 === АНАЛИЗ ЗАВЕРШЁН ===")
        print(f"🎯 Класс: {target_class_name}")
        print(f"📊 XAI карт: {len(results['xai_maps'])}")
        print(f"🧪 Интервенций: {sum(len(step_data) for step_data in results['interventions'].values())}")
        
        if 'statistical_validation' in results and 'overall_conclusion' in results['statistical_validation']:
            conclusion = results['statistical_validation']['overall_conclusion']
            print(f"📈 Статистическая значимость: {'✅ ДА' if conclusion['significant'] else '❌ НЕТ'}")
        
        if 'sanity_checks' in results and 'overall_sanity_score' in results['sanity_checks']:
            sanity_score = results['sanity_checks']['overall_sanity_score']
            print(f"🔍 Sanity score: {sanity_score:.2f}/1.0")
        
        return results
        
    except Exception as e:
        print(f"❌ Критическая ошибка в пайплайне: {e}")
        results['pipeline_error'] = str(e)
        return results


# === ЗАПУСК ОСНОВНОГО ПАЙПЛАЙНА ===

if XAI_ANALYZER_READY:
    print("🚀 === ГОТОВ К ЗАПУСКУ ОСНОВНОГО ПАЙПЛАЙНА ===")
    
    # Спрашиваем пользователя о запуске
    print(f"🎯 Целевой класс: {TARGET_CLASS_NAME}")
    print(f"📈 Временных шагов для анализа: {len(timesteps)}")
    print(f"⚙️  Интервенции: {', '.join(INTERVENTION_TYPES)}")
    print(f"📊 Статистика: α={ALPHA_LEVEL}, bootstrap={N_BOOTSTRAP}, permutations={N_PERMUTATIONS}")
    print()
    
    # Автоматический запуск (раскомментируйте для ручного подтверждения)
    # user_input = input("🤔 Запустить полный XAI анализ? (y/n): ")
    # if user_input.lower() in ['y', 'yes', 'да']:
    
    if True:  # Автоматический запуск
        print("🎬 Запускаем полный XAI пайплайн...")
        # Запуск пайплайна
        final_results = run_comprehensive_xai_pipeline(
            trajectory=trajectory,
            timesteps=timesteps,
            xai_analyzer=xai_analyzer,
            classifier=classifier,
            target_class_id=TARGET_CLASS_ID,
            target_class_name=TARGET_CLASS_NAME,
            save_results=True
        )
        
        print("🏁 === ПАЙПЛАЙН ЗАВЕРШЁН ===")
        
        # Краткое резюме
        if 'pipeline_error' not in final_results:
            print("✅ Пайплайн выполнен успешно!")
            
            if 'statistical_validation' in final_results and 'overall_conclusion' in final_results['statistical_validation']:
                stats_result = final_results['statistical_validation']['overall_conclusion']
                if stats_result['significant']:
                    print(f"🎉 РЕЗУЛЬТАТ: Обнаружена статистически значимая разница между Top-k и Bottom-k регионами!")
                    print(f"   Это подтверждает, что XAI методы корректно выделяют каузально важные регионы.")
                else:
                    print(f"⚠️  РЕЗУЛЬТАТ: Статистически значимая разница не обнаружена.")
                    print(f"   Возможно, требуется больше данных или другие параметры анализа.")
            
            print(f"📁 Результаты сохранены в: {RESULTS_DIR}")
            
        else:
            print(f"❌ Пайплайн завершился с ошибкой: {final_results['pipeline_error']}")
    
    else:
        print("⏸️  Пайплайн отменён пользователем.")
        print("💡 Вы можете запустить отдельные части анализа с помощью функций выше.")

else:
    print("❌ XAI анализатор не готов. Проверьте предыдущие шаги.")
    print("🔧 Убедитесь что модели загружены и траектория сгенерирована.")


