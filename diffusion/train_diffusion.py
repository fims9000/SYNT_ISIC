import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import gc
from torch.cuda import amp

# Установим переменную окружения для оптимизации памяти
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# --- ФИНАЛЬНАЯ функция коррекции цвета ---
def enhance_color(img, class_id):
    """Адаптивная коррекция цвета с индивидуальными параметрами для каждого класса"""
    img_array = np.array(img, dtype=np.float32) / 255.0
    current_mean = np.mean(img_array, axis=(0, 1))

    class_params = {
        # NV
        0: {'gain': [1.04462, 0.8474, 0.7931], 'brightness': 0.23741, 'target': [0.7525, 0.5645, 0.5303]},
        # MEL
        1: {'gain': [1.0561, 0.86, 0.883], 'brightness': 0.218, 'target': [0.7453, 0.54, 0.5721]},
        # BCC
        2: {'gain': [1.125, 0.99, 0.922], 'brightness': 0.262, 'target': [0.784, 0.635, 0.573]},
        # AKIEC
        3: {'gain': [1.158, 0.952, 0.82], 'brightness': 0.275, 'target': [0.781, 0.618, 0.593]},
        # BKL
        4: {'gain': [1.1242, 0.846, 0.796], 'brightness': 0.25, 'target': [0.766, 0.574, 0.561]},
        # DF
        5: {'gain': [1.0, 1.1, 1.1], 'brightness': 0.23, 'target': [0.79, 0.66, 0.66]},
        # VASC
        6: {'gain': [1.08, 1.05, 0.945], 'brightness': 0.09, 'target': [0.79, 0.64, 0.597]}
    }

    params = class_params[class_id]
    gain = params['gain']
    brightness_boost = params['brightness']
    target_mean = params['target']

    for c in range(3):
        diff = target_mean[c] - current_mean[c]
        img_array[..., c] = np.clip(img_array[..., c] + diff * gain[c] + brightness_boost, 0, 1)

    return Image.fromarray((img_array * 255).astype(np.uint8))


# --- Конфигурация ---
BASE_DIR = os.path.expanduser('~/MaxYura')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ISIC2018_Task3_Training_Input')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'ISIC2018_Task3_Training_GroundTruth.csv')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
IMAGE_SIZE = 128
BATCH_SIZE = 2
NUM_CLASSES = 7
LR = 1e-4
TIMESTEPS = 1000
MAX_SAMPLES_PER_CLASS = 500
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 50

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# --- Улучшенные трансформации ---
transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.RandomApply([transforms.CenterCrop(IMAGE_SIZE)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# --- Класс датасета с индивидуальной коррекцией ---
class SingleClassDataset(Dataset):
    def __init__(self, image_dir, csv_path, class_id, image_size=128, transform=None, max_samples=500):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.class_id = class_id

        df = pd.read_csv(csv_path)
        classes = [col for col in df.columns if col != 'image']
        df['label'] = df[classes].values.argmax(axis=1)

        available = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')}
        df = df[df['image'].isin(available)].reset_index(drop=True)

        class_data = df[df['label'] == class_id]
        num_samples = min(max_samples, len(class_data))
        self.data = class_data.sample(n=num_samples, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'] + '.jpg')
        img = Image.open(img_path).convert('RGB').resize((self.image_size, self.image_size))

        # Применяем индивидуальную коррекцию цвета
        img = enhance_color(img, self.class_id)

        return self.transform(img) if self.transform else transforms.ToTensor()(img)


# --- Оптимизированная модель с Attention ---
def create_model():
    return UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
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


# --- Интерактивный выбор классов ---
def select_classes():
    selected_classes = []
    
    print("\n" + "="*50)
    print("Выберите классы для обучения (введите номер):")
    print("="*50)
    
    # Выводим меню классов
    for i, name in enumerate(CLASS_NAMES, 1):
        print(f"{i} - {name}")
    print("8 - Начать обучение")
    print("0 - Очистить выбор")
    print("="*50)
    
    while True:
        # Показываем текущий выбор
        if selected_classes:
            selected_names = [CLASS_NAMES[idx] for idx in selected_classes]
            print(f"\nТекущий выбор: {', '.join(selected_names)}")
        else:
            print("\nТекущий выбор: нет выбранных классов")
        
        # Запрос ввода
        choice = input("Введите номер класса или команду (8 - начать обучение): ").strip()
        
        try:
            choice_num = int(choice)
            
            # Обработка команд
            if choice_num == 8:
                if not selected_classes:
                    print("Ошибка: не выбрано ни одного класса!")
                    continue
                return selected_classes
            elif choice_num == 0:
                selected_classes = []
                print("Выбор очищен!")
                continue
            
            # Проверка диапазона
            if choice_num < 1 or choice_num > len(CLASS_NAMES):
                print(f"Ошибка: введите число от 1 до {len(CLASS_NAMES)} или 8 для запуска")
                continue
            
            # Вычисляем индекс класса
            class_idx = choice_num - 1
            
            # Проверяем, не выбран ли уже класс
            if class_idx in selected_classes:
                print(f"Класс {CLASS_NAMES[class_idx]} уже выбран!")
            else:
                selected_classes.append(class_idx)
                print(f"Добавлен класс: {CLASS_NAMES[class_idx]}")
        
        except ValueError:
            print("Ошибка: введите число!")


# --- Обучение с сохранением лучшей модели ---
def train_class(class_id, class_name):
    print(f"\n=== Обучение для класса {class_name} ({class_id}) ===")

    dataset = SingleClassDataset(
        DATA_DIR,
        CSV_PATH,
        class_id,
        IMAGE_SIZE,
        transform,
        MAX_SAMPLES_PER_CLASS
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    print(f"Используется изображений: {len(dataset)}/{MAX_SAMPLES_PER_CLASS}")

    model = create_model().to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS, beta_schedule="squaredcos_cap_v2")
    optimizer = Adam(model.parameters(), lr=LR)
    scaler = amp.GradScaler()

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for images in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
                noisy_images = scheduler.add_noise(images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            del images, noise, noisy_images, noise_pred, loss
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(loader)
        print(f"Loss: {avg_loss:.5f}")

        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_{class_name}_best.pth"))
            print(f"Сохранена лучшая модель (loss: {best_loss:.5f})")

        # Сохраняем чекпоинты каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_{class_name}_epoch_{epoch + 1:02d}.pth"))

        torch.cuda.empty_cache()
        gc.collect()

    del model, optimizer, scheduler, scaler, dataset, loader
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Обучение класса {class_name} завершено\n")


# --- Главный цикл программы ---
if __name__ == "__main__":
    # Получаем список классов для обучения
    selected_classes = select_classes()
    
    # Обучаем выбранные классы
    for class_idx in selected_classes:
        class_name = CLASS_NAMES[class_idx]
        train_class(class_idx, class_name)
    
    print("\nОбучение всех выбранных классов завершено!")