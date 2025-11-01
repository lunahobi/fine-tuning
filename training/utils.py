"""
Утилиты для обучения и воспроизводимости
"""

import random
import numpy as np
import torch
import os
import platform
import psutil

def set_random_seeds(seed: int = 42):
    """
    Устанавливает все генераторы случайных чисел для обеспечения воспроизводимости
    
    Args:
        seed: Значение seed для всех генераторов
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Дополнительные настройки для PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Установка переменной окружения для дополнительной воспроизводимости
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Установлен seed: {seed}")
    print("Воспроизводимость настроена для:")
    print("- Python random")
    print("- NumPy")
    print("- PyTorch (CPU и GPU)")
    print("- CuDNN")

def get_device(device_name: str = "auto"):
    """
    Определяет доступное устройство для обучения
    
    Args:
        device_name: Название устройства ("auto", "cpu", "cuda")
    
    Returns:
        torch.device: Устройство для обучения
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Используется CPU")
    else:
        device = torch.device(device_name)
        print(f"Используется устройство: {device}")
    
    return device

def print_system_info():
    """Выводит информацию о системе"""
    print("=" * 50)
    print("ИНФОРМАЦИЯ О СИСТЕМЕ")
    print("=" * 50)
    
    # Операционная система
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Архитектура: {platform.machine()}")
    print(f"Процессор: {platform.processor()}")
    
    # Python
    print(f"Python: {platform.python_version()}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Память
    memory = psutil.virtual_memory()
    print(f"ОЗУ: {memory.total / (1024**3):.1f} GB")
    print(f"Доступно ОЗУ: {memory.available / (1024**3):.1f} GB")
    
    print("=" * 50)

def count_parameters(model):
    """
    Подсчитывает количество параметров модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        dict: Словарь с количеством параметров
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Сохраняет чекпоинт модели
    
    Args:
        model: PyTorch модель
        optimizer: Оптимизатор
        epoch: Номер эпохи
        loss: Значение потерь
        accuracy: Точность
        filepath: Путь для сохранения
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    torch.save(checkpoint, filepath)
    print(f"Чекпоинт сохранен: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Загружает чекпоинт модели
    
    Args:
        filepath: Путь к файлу чекпоинта
        model: PyTorch модель
        optimizer: Оптимизатор (опционально)
    
    Returns:
        dict: Данные чекпоинта
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Чекпоинт загружен: {filepath}")
    print(f"Эпоха: {checkpoint['epoch']}")
    print(f"Потери: {checkpoint['loss']:.4f}")
    print(f"Точность: {checkpoint['accuracy']:.4f}")
    
    return checkpoint







