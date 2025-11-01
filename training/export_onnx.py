"""
Скрипт для экспорта обученной модели в ONNX формат
"""

import argparse
import torch
import torch.onnx
from pathlib import Path
import timm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from training.config import Config
from training.utils import get_device, set_random_seeds

def export_to_onnx(model_path: Path, output_path: Path, config: Config):
    """
    Экспортирует PyTorch модель в ONNX формат
    
    Args:
        model_path: Путь к сохраненной PyTorch модели
        output_path: Путь для сохранения ONNX модели
        config: Конфигурация модели
    """
    print(f"Экспорт модели в ONNX: {model_path} -> {output_path}")
    
    # Загружаем модель
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Создаем модель с той же архитектурой
    model = timm.create_model(
        config.model.model_name,
        pretrained=False,
        num_classes=config.data.num_classes
    )
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем пример входных данных
    dummy_input = torch.randn(1, 3, *config.data.image_size)
    
    # Экспортируем в ONNX
    torch.onnx.export(
        model,                          # модель
        dummy_input,                    # пример входных данных
        output_path,                    # путь для сохранения
        export_params=True,             # сохранить веса модели
        opset_version=9,                # версия ONNX opset (совместимая с ONNX Runtime 1.16.3)
        do_constant_folding=True,       # оптимизация констант
        input_names=['input'],          # имена входных тензоров
        output_names=['output'],        # имена выходных тензоров
        dynamic_axes={                  # динамические оси
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Модель успешно экспортирована в ONNX: {output_path}")
    
    # Проверяем размер файла
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Размер ONNX файла: {file_size:.2f} MB")

def verify_onnx_model(onnx_path: Path, config: Config):
    """
    Проверяет корректность экспортированной ONNX модели
    
    Args:
        onnx_path: Путь к ONNX модели
        config: Конфигурация модели
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"\nПроверка ONNX модели: {onnx_path}")
        
        # Проверка структуры модели
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ Структура модели корректна")
        
        # Проверка выполнения инференса
        session = ort.InferenceSession(onnx_path)
        
        # Получаем информацию о входных и выходных данных
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"✓ Входные данные: {input_info.name}, форма: {input_info.shape}")
        print(f"✓ Выходные данные: {output_info.name}, форма: {output_info.shape}")
        
        # Тестовый инференс
        import numpy as np
        test_input = np.random.randn(1, 3, *config.data.image_size).astype(np.float32)
        outputs = session.run([output_info.name], {input_info.name: test_input})
        
        print(f"✓ Тестовый инференс успешен, выходная форма: {outputs[0].shape}")
        print("✓ ONNX модель готова к использованию!")
        
        return True
        
    except ImportError:
        print("⚠ onnx или onnxruntime не установлены. Пропускаем проверку.")
        return False
    except Exception as e:
        print(f"✗ Ошибка при проверке ONNX модели: {e}")
        return False

def main():
    """Основная функция экспорта"""
    parser = argparse.ArgumentParser(description='Экспорт модели в ONNX')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Путь к обученной PyTorch модели')
    parser.add_argument('--output_path', type=str, 
                       help='Путь для сохранения ONNX модели')
    parser.add_argument('--config_path', type=str,
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    # Создаем конфигурацию
    if args.config_path:
        config = Config.load_config(Path(args.config_path))
    else:
        config = Config()
    
    # Устанавливаем воспроизводимость
    set_random_seeds(config.experiment.random_seed)
    
    # Определяем пути
    model_path = Path(args.model_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    
    # Проверяем существование модели
    if not model_path.exists():
        print(f"Ошибка: Модель не найдена: {model_path}")
        return
    
    # Создаем выходную директорию
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Экспортируем модель
        export_to_onnx(model_path, output_path, config)
        
        # Проверяем экспортированную модель
        verify_onnx_model(output_path, config)
        
        print(f"\nЭкспорт завершен успешно!")
        print(f"ONNX модель сохранена: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при экспорте: {e}")
        return

if __name__ == "__main__":
    main()










