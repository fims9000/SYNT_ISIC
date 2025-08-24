#!/usr/bin/env python3
"""
Скрипт для автоматической загрузки моделей ISIC с Google Drive
Автоматически создает папку checkpoints/ и загружает все необходимые модели
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file_from_drive(file_id, output_path, chunk_size=8192):
    """Загружает файл с Google Drive по ID"""
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Ошибка при загрузке {output_path.name}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Распаковывает ZIP архив"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Распакован архив: {zip_path.name}")
        return True
    except Exception as e:
        print(f"Ошибка при распаковке {zip_path.name}: {e}")
        return False

def main():
    print("🚀 Загрузка моделей ISIC с Google Drive")
    print("=" * 50)
    
    # Создаем папку checkpoints если её нет
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    # ID файлов на Google Drive
    models_to_download = {
        "models_checkpoints.zip": "1kTIHp98AuvLmee5LahH-hZQqXQik_B1U"  # ID файла с Google Drive
    }
    
    print("📁 Папка checkpoints/ создана/найдена")
    print(f"📥 Будет загружено файлов: {len(models_to_download)}")
    print()
    
    # Загружаем каждый файл
    for filename, file_id in models_to_download.items():
        output_path = checkpoints_dir / filename
        
        print(f"📥 Загружаю: {filename}")
        if download_file_from_drive(file_id, output_path):
            print(f"✓ Загружен: {filename}")
            
            # Если это ZIP архив, распаковываем
            if filename.endswith('.zip'):
                print(f"📦 Распаковываю: {filename}")
                if extract_zip(output_path, checkpoints_dir):
                    # Удаляем ZIP после распаковки
                    output_path.unlink()
                    print(f"🗑️ Удален архив: {filename}")
        else:
            print(f"❌ Ошибка загрузки: {filename}")
            return False
    
    print()
    print("🎉 Загрузка завершена!")
    print("📁 Модели находятся в папке: checkpoints/")
    print()
    print("📋 Содержимое папки checkpoints/:")
    
    # Показываем содержимое
    for item in checkpoints_dir.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.1f} МБ)")
        elif item.is_dir():
            print(f"  📁 {item.name}/")
    
    print()
    print("✅ Теперь можно запускать ISICGUI!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Загрузка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)
