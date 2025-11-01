"""
Простой скрипт для экспорта модели в ONNX
Работает из корневой директории проекта fine-tuning
"""

import sys
import os
from pathlib import Path
import torch
import torch.onnx
import timm
import json

# Устанавливаем переменные окружения для отключения эмодзи и обработки кодировки
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TORCH_ONNX_DISABLE_EMOJI'] = '1'

# Настраиваем stdout для Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def export_model_to_onnx(model_path=None, output_path=None):
    """Экспортирует обученную модель в ONNX формат"""
    
    # Пути к файлам (относительно корня проекта)
    if model_path is None:
        model_path = Path("models/best_resnet18.pth")
    else:
        model_path = Path(model_path)
        
    if output_path is None:
        output_path = Path("models/resnet18_final.onnx")
    else:
        output_path = Path(output_path)
    
    print("Экспорт модели в ONNX...")
    print(f"Модель: {model_path}")
    print(f"Выходной файл: {output_path}")
    
    # Проверяем существование модели
    if not model_path.exists():
        print(f"Ошибка: Модель не найдена: {model_path}")
        print(f"Текущая директория: {Path.cwd()}")
        print("Содержимое директории:")
        for item in Path.cwd().iterdir():
            print(f"   - {item.name}")
        return False
    
    try:
        # Загружаем модель
        print("Загрузка модели...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Определяем архитектуру модели из конфигурации
        config_dict = checkpoint.get('config', {})
        model_name = config_dict.get('model_name', 'resnet18')
        classes = config_dict.get('classes', ['apple', 'kiwi', 'mandarin'])
        num_classes = len(classes)
        
        # Получаем метрики из checkpoint, если есть
        val_acc = checkpoint.get('val_acc', None)
        
        print(f"Создание модели: {model_name} с {num_classes} классами")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Создаем пример входных данных
        print("Подготовка к экспорту...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Экспортируем в ONNX (используем старый API для совместимости)
        print("Экспорт в ONNX...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подавляем вывод эмодзи от torch.onnx через перенаправление stdout
        import io
        import contextlib
        
        # Создаем заглушку для stdout при экспорте, чтобы избежать ошибок кодировки
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=9,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        # Проверяем размер файла
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print("Модель успешно экспортирована!")
        print(f"Размер ONNX файла: {file_size:.2f} MB")
        
        # Проверяем ONNX модель
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("ONNX модель проверена успешно!")
        except ImportError:
            print("ONNX не установлен, пропускаем проверку")
        except Exception as e:
            print(f"Предупреждение: проверка ONNX не удалась: {e}")
        
        # Создаем информацию о модели на основе реальных данных
        model_info = {
            "model_name": model_name,
            "num_classes": num_classes,
            "classes": classes,
            "input_size": [224, 224],
            "onnx_path": str(output_path.as_posix()),
            "pytorch_file": str(model_path.as_posix())
        }
        
        # Добавляем метрики, если есть
        if val_acc is not None:
            model_info["val_accuracy"] = float(val_acc)
        
        # Сохраняем информацию
        info_path = Path("models/model_info.json")
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"Информация о модели сохранена: {info_path}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при экспорте: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Экспорт модели в ONNX')
    parser.add_argument('--model_path', type=str, help='Путь к модели .pth')
    parser.add_argument('--output_path', type=str, help='Путь для сохранения ONNX')
    
    args = parser.parse_args()
    
    success = export_model_to_onnx(args.model_path, args.output_path)
    if success:
        print("\nЭкспорт завершен успешно!")
        print("ONNX модель готова для использования в приложениях!")
    else:
        print("\nЭкспорт не удался!")
        sys.exit(1)










