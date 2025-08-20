import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.expanduser('~/MaxYura')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ImagesTest')
IMAGE_SIZE = 128
TRAIN_TIMESTEPS = 1000  # Должно совпадать с обучением
INFERENCE_TIMESTEPS = 1000  # Можно уменьшить для ускорения генерации
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLASS = 4

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
EPOCH_INPUT = 45
CHECKPOINT_MODE = "best"  # "best" или "epoch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_id, class_name in enumerate(CLASS_NAMES):
    if CHECKPOINT_MODE == "best":
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_{class_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Нет лучшего чекпоинта для класса {class_name}")
            continue
        print(f"Класс {class_name}: загружаем ЛУЧШУЮ модель")
    else:
        files = os.listdir(CHECKPOINT_DIR)
        epochs = []
        for f in files:
            if f.startswith(f"unet_{class_name}_epoch_") and f.endswith(".pth"):
                try:
                    epoch_num = int(f.split("_epoch_")[1].split(".")[0])
                    epochs.append(epoch_num)
                except Exception:
                    continue
        if not epochs:
            print(f"Нет чекпоинтов для {class_name}")
            continue

        best_epoch = max(e for e in epochs if e <= EPOCH_INPUT)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_{class_name}_epoch_{best_epoch:02d}.pth")
        print(f"Класс {class_name}: загружаем чекпоинт эпохи {best_epoch}")

    # Идентичная архитектура с обучением
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
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
    ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Настройка scheduler идентичная обучению
    scheduler = DDPMScheduler(
        num_train_timesteps=TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    
    scheduler.set_timesteps(INFERENCE_TIMESTEPS, device=DEVICE)

    class_out_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_out_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(SAMPLES_PER_CLASS):
            sample = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

            for t in tqdm(scheduler.timesteps, desc=f"Генерация {class_name} {i + 1}/{SAMPLES_PER_CLASS}"):
                noise_pred = model(sample, t).sample
                sample = scheduler.step(noise_pred, t, sample).prev_sample

            # Обратное преобразование нормализации
            image = sample.clamp(-1, 1)
            image = (image + 1) * 0.5
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)

            img_pil = Image.fromarray(image)
            img_pil.save(os.path.join(class_out_dir, f"{class_name}_{i + 1:02d}.png"))
            print(f"Сохранено: {class_name}_{i + 1:02d}.png")

            del sample, image, img_pil
            torch.cuda.empty_cache()

print("Генерация завершена.")