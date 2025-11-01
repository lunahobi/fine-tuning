"""
Скрипт для подготовки и разделения датасета фруктов
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import pandas as pd

# Конфигурация
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
CLASSES = ["apple", "kiwi", "mandarin"]

# Параметры разделения
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def create_directory_structure():
    """Создает структуру директорий"""
    print("Создание структуры директорий...")
    
    # Создаем основные директории
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Создаем поддиректории для каждого класса
    for split in ['train', 'val', 'test']:
        for class_name in CLASSES:
            (PROCESSED_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("УСПЕХ: Структура директорий создана")

def validate_images(data_dir):
    """Проверяет и валидирует изображения"""
    print(f"Проверка изображений в {data_dir}...")
    
    corrupted_files = []
    valid_files = []
    
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"ПРЕДУПРЕЖДЕНИЕ Директория {class_dir} не существует")
            continue
            
        for img_file in class_dir.glob("*.webp"):
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Проверяем целостность
                valid_files.append(img_file)
            except Exception as e:
                print(f"ОШИБКА Поврежденный файл: {img_file} - {e}")
                corrupted_files.append(img_file)
    
    print(f"УСПЕХ Найдено валидных изображений: {len(valid_files)}")
    print(f"ОШИБКА Поврежденных изображений: {len(corrupted_files)}")
    
    return valid_files, corrupted_files

def split_dataset(raw_data_dir, processed_data_dir, classes, train_split, val_split, test_split, random_seed=42):
    """Разделяет датасет на train/val/test"""
    print("ДАННЫЕ Разделение датасета...")
    
    # Устанавливаем seed для воспроизводимости
    random.seed(random_seed)
    
    # Создаем структуру директорий
    create_directory_structure()
    
    dataset_info = []
    
    for class_name in classes:
        class_dir = raw_data_dir / class_name
        if not class_dir.exists():
            print(f"ПРЕДУПРЕЖДЕНИЕ Директория {class_dir} не существует")
            continue
        
        # Получаем все изображения
        image_files = list(class_dir.glob("*.webp"))
        print(f"ИЗОБРАЖЕНИЙ Найдено изображений {class_name}: {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"ОШИБКА Нет изображений в {class_name}")
            continue
        
        # Перемешиваем файлы
        random.shuffle(image_files)
        
        # Вычисляем индексы разделения
        total_files = len(image_files)
        train_end = int(total_files * train_split)
        val_end = train_end + int(total_files * val_split)
        
        # Разделяем файлы
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        print(f"  - Train: {len(train_files)}")
        print(f"  - Val: {len(val_files)}")
        print(f"  - Test: {len(test_files)}")
        
        # Копируем файлы
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in files:
                dest_path = processed_data_dir / split / class_name / img_file.name
                shutil.copy2(img_file, dest_path)
                
                # Записываем информацию
                dataset_info.append({
                    'original_path': str(img_file),
                    'new_path': str(dest_path),
                    'class': class_name,
                    'split': split,
                    'filename': img_file.name
                })
    
    # Сохраняем информацию о датасете
    df = pd.DataFrame(dataset_info)
    df.to_csv(processed_data_dir / "dataset_info.csv", index=False)
    
    print(f"УСПЕХ Разделение завершено")
    print(f"ДАННЫЕ Информация сохранена в {processed_data_dir / 'dataset_info.csv'}")
    
    return dataset_info

def main():
    """Основная функция"""
    print("Подготовка датасета фруктов")
    print("=" * 50)
    
    # Проверяем наличие исходных данных
    if not Path("data/raw").exists():
        print("ОШИБКА: Директория data/raw не существует!")
        print("Убедитесь, что у вас есть изображения фруктов в папках:")
        print("   - data/raw/apple/")
        print("   - data/raw/kiwi/")
        print("   - data/raw/mandarin/")
        return
    
    # Валидируем изображения
    valid_files, corrupted_files = validate_images(Path("data/raw"))
    
    if len(valid_files) == 0:
        print("ОШИБКА: Нет валидных изображений для обработки!")
        return
    
    # Разделяем датасет
    dataset_info = split_dataset(
        Path("data/raw"),
        Path("data/processed"),
        CLASSES,
        TRAIN_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT
    )
    
    print("\nУСПЕХ: Подготовка данных завершена!")
    print(f"Обработанные данные: data/processed/")
    print(f"Всего файлов обработано: {len(dataset_info)}")

if __name__ == "__main__":
    main()
