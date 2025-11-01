#!/usr/bin/env python3
"""
Скрипт для выбора лучшей модели
"""

import subprocess
import sys
try:
    # Попытка печати в UTF-8 (Windows консоль часто cp1251)
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

def train_model(model_name, epochs=20, lr=0.001, batch_size=32):
    """Ищет уже обученную модель или обучает новую"""
    print(f"\nПРОВЕРКА Проверяем наличие обученной модели {model_name}...")
    print("=" * 60)
    
    # Сначала ищем уже обученную модель
    existing_model_dir = find_latest_experiment(model_name)
    
    if existing_model_dir:
        print(f"УСПЕХ Используем уже обученную модель {model_name}")
        print(f"Директория: {existing_model_dir}")
        return True, 0  # Время обучения = 0, так как модель уже готова
    
    # Если модель не найдена, обучаем новую
    print(f"ОБУЧЕНИЕ Модель {model_name} не найдена, начинаем обучение...")
    start_time = time.time()
    
    try:
        # Запускаем обучение
        result = subprocess.run([
            sys.executable, "training/train_with_plots.py",
            "--model", model_name,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--device", "cpu"
        ], capture_output=True, text=True, cwd=".")
        
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"УСПЕХ {model_name} обучена успешно!")
            print(f"Время обучения: {training_time/60:.2f} минут")
            return True, training_time
        else:
            print(f"ОШИБКА Ошибка при обучении {model_name}:")
            print(result.stderr)
            return False, training_time
            
    except Exception as e:
        print(f"ОШИБКА Исключение при обучении {model_name}: {e}")
        return False, time.time() - start_time

def find_latest_experiment(model_name):
    """Находит последний эксперимент для модели с файлом results.json"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None
    
    # Ищем папки с именем модели
    model_dirs = [d for d in experiments_dir.iterdir() 
                  if d.is_dir() and model_name in d.name]
    
    if not model_dirs:
        return None
    
    # Фильтруем только те директории, где есть results.json
    valid_dirs = []
    for d in model_dirs:
        results_file = d / "results.json"
        if results_file.exists():
            valid_dirs.append(d)
    
    if not valid_dirs:
        print(f"ПРЕДУПРЕЖДЕНИЕ Не найдены директории с results.json для {model_name}")
        return None
    
    # Возвращаем самую новую папку с results.json
    return max(valid_dirs, key=lambda x: x.stat().st_mtime)

def load_model_results(experiment_dir):
    """Загружает результаты модели"""
    results_file = experiment_dir / "results.json"
    if not results_file.exists():
        print(f"ОШИБКА Файл результатов не найден: {results_file}")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"УСПЕХ Результаты загружены из: {results_file}")
        return results
    except json.JSONDecodeError as e:
        print(f"ОШИБКА Ошибка JSON в файле {results_file}: {e}")
        return None
    except Exception as e:
        print(f"ОШИБКА Ошибка при чтении файла {results_file}: {e}")
        return None

def compare_models():
    """Сравнивает результаты обеих моделей (последние эксперименты)"""
    print("\nДАННЫЕ Сравнение моделей (последние эксперименты)...")
    print("=" * 60)
    print("Ищем ПОСЛЕДНИЕ эксперименты для каждой модели по времени модификации")
    print("=" * 60)
    
    # Находим последние эксперименты
    resnet_dir = find_latest_experiment("resnet18")
    efficientnet_dir = find_latest_experiment("efficientnet_b0")
    
    if resnet_dir:
        print(f"[OK] ResNet18: {resnet_dir.name}")
        try:
            print(f"   Время модификации: {time.ctime(resnet_dir.stat().st_mtime)}")
        except:
            pass
    else:
        print("[ERROR] ResNet18: не найден")
    
    if efficientnet_dir:
        print(f"[OK] EfficientNet-B0: {efficientnet_dir.name}")
        try:
            print(f"   Время модификации: {time.ctime(efficientnet_dir.stat().st_mtime)}")
        except:
            pass
    else:
        print("[ERROR] EfficientNet-B0: не найден")
    
    if not resnet_dir and not efficientnet_dir:
        print("ОШИБКА Не найдены результаты экспериментов")
        print("   - ResNet18 не найден")
        print("   - EfficientNet-B0 не найден")
        return None
    
    # Если отсутствует одна модель, работаем только с доступной
    if not resnet_dir:
        print("[WARN] ResNet18 не найден, используем только EfficientNet-B0")
        efficientnet_results = load_model_results(efficientnet_dir)
        if not efficientnet_results:
            return None
        return "EfficientNet-B0", efficientnet_dir, efficientnet_results
    
    if not efficientnet_dir:
        print("[WARN] EfficientNet-B0 не найден, используем только ResNet18")
        resnet_results = load_model_results(resnet_dir)
        if not resnet_results:
            return None
        return "ResNet18", resnet_dir, resnet_results
    
    # Загружаем результаты
    resnet_results = load_model_results(resnet_dir)
    efficientnet_results = load_model_results(efficientnet_dir)
    
    if not resnet_results or not efficientnet_results:
        print("ОШИБКА Не удалось загрузить результаты")
        return None
    
    # Создаем таблицу сравнения
    comparison_data = {
        'ResNet18': {
            'Test Accuracy': f"{resnet_results['test_accuracy']:.4f}",
            'Val Accuracy': f"{resnet_results['best_val_accuracy']:.2f}%",
            'Parameters': f"{resnet_results['model_info']['total_parameters']:,}",
            'Model Size (MB)': f"{resnet_results['model_info']['model_size_mb']:.1f}",
            'Training Time (min)': f"{resnet_results['training_time_minutes']:.1f}",
            'Experiment Dir': str(resnet_dir)
        },
        'EfficientNet-B0': {
            'Test Accuracy': f"{efficientnet_results['test_accuracy']:.4f}",
            'Val Accuracy': f"{efficientnet_results['best_val_accuracy']:.2f}%",
            'Parameters': f"{efficientnet_results['model_info']['total_parameters']:,}",
            'Model Size (MB)': f"{efficientnet_results['model_info']['model_size_mb']:.1f}",
            'Training Time (min)': f"{efficientnet_results['training_time_minutes']:.1f}",
            'Experiment Dir': str(efficientnet_dir)
        }
    }
    
    df_comparison = pd.DataFrame(comparison_data).T
    
    print("Сравнение результатов:")
    print(df_comparison)
    
    # Определяем лучшую модель
    # ВАЖНО: Используем val_accuracy как основной критерий, так как:
    # 1. Test set должен оставаться "чистым" для финальной оценки
    # 2. Val accuracy используется для выбора модели во время обучения
    # 3. Использование test для выбора модели может привести к data leakage
    
    resnet_val_acc = resnet_results['best_val_accuracy'] / 100.0  # Конвертируем из процентов
    efficientnet_val_acc = efficientnet_results['best_val_accuracy'] / 100.0
    
    resnet_test_acc = resnet_results['test_accuracy']
    efficientnet_test_acc = efficientnet_results['test_accuracy']
    
    resnet_size = resnet_results['model_info']['model_size_mb']
    efficientnet_size = efficientnet_results['model_info']['model_size_mb']
    
    print(f"\nАНАЛИЗ Выбор лучшей модели:")
    print(f"   ResNet18:      val_acc={resnet_val_acc:.4f}, test_acc={resnet_test_acc:.4f}, size={resnet_size:.2f} MB")
    print(f"   EfficientNet-B0: val_acc={efficientnet_val_acc:.4f}, test_acc={efficientnet_test_acc:.4f}, size={efficientnet_size:.2f} MB")
    print(f"\n   Критерий выбора: val_accuracy (test set используется только для финальной оценки)")
    
    selection_reason = ""
    
    # Сначала сравниваем по val_accuracy (основной критерий)
    if resnet_val_acc > efficientnet_val_acc:
        best_model = "ResNet18"
        best_dir = resnet_dir
        best_results = resnet_results
        selection_reason = f"ResNet18 имеет большую val_accuracy ({resnet_val_acc:.4f} vs {efficientnet_val_acc:.4f})"
    elif efficientnet_val_acc > resnet_val_acc:
        best_model = "EfficientNet-B0"
        best_dir = efficientnet_dir
        best_results = efficientnet_results
        selection_reason = f"EfficientNet-B0 имеет большую val_accuracy ({efficientnet_val_acc:.4f} vs {resnet_val_acc:.4f})"
    else:
        # Если val_accuracy одинаковая, сравниваем по test_accuracy
        print(f"   ИНФОРМАЦИЯ: val_accuracy одинаковая ({resnet_val_acc:.4f}), сравниваем по test_accuracy")
        if resnet_test_acc > efficientnet_test_acc:
            best_model = "ResNet18"
            best_dir = resnet_dir
            best_results = resnet_results
            selection_reason = f"val_accuracy одинаковая ({resnet_val_acc:.4f}), ResNet18 имеет большую test_accuracy ({resnet_test_acc:.4f} vs {efficientnet_test_acc:.4f})"
        elif efficientnet_test_acc > resnet_test_acc:
            best_model = "EfficientNet-B0"
            best_dir = efficientnet_dir
            best_results = efficientnet_results
            selection_reason = f"val_accuracy одинаковая ({resnet_val_acc:.4f}), EfficientNet-B0 имеет большую test_accuracy ({efficientnet_test_acc:.4f} vs {resnet_test_acc:.4f})"
        else:
            # Если обе точности одинаковые, выбираем по размеру модели
            print(f"   ИНФОРМАЦИЯ: val и test accuracy одинаковые, выбираем по размеру модели")
            if resnet_size < efficientnet_size:
                best_model = "ResNet18"
                best_dir = resnet_dir
                best_results = resnet_results
                selection_reason = f"val и test accuracy одинаковые, ResNet18 меньше по размеру ({resnet_size:.2f} MB vs {efficientnet_size:.2f} MB)"
            else:
                best_model = "EfficientNet-B0"
                best_dir = efficientnet_dir
                best_results = efficientnet_results
                selection_reason = f"val и test accuracy одинаковые, EfficientNet-B0 меньше по размеру ({efficientnet_size:.2f} MB vs {resnet_size:.2f} MB)"
    
    print(f"\nПОБЕДИТЕЛЬ Лучшая модель: {best_model}")
    print(f"ПРИЧИНА: {selection_reason}")
    print(f"Директория: {best_dir}")
    print(f"Test Accuracy: {best_results['test_accuracy']:.4f}")
    print(f"Val Accuracy: {best_results['best_val_accuracy']:.2f}%")
    print(f"Размер модели: {best_results['model_info']['model_size_mb']:.2f} MB")
    
    # Создаем визуализацию сравнения
    create_comparison_plots(resnet_results, efficientnet_results, best_model)
    
    return best_model, best_dir, best_results

def create_comparison_plots(resnet_results, efficientnet_results, best_model):
    """Создает графики сравнения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = ['ResNet18', 'EfficientNet-B0']
    colors = ['skyblue', 'lightcoral']
    
    # 1. Test Accuracy
    accuracies = [resnet_results['test_accuracy'], efficientnet_results['test_accuracy']]
    bars1 = axes[0,0].bar(models, accuracies, color=colors)
    axes[0,0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_ylim(0, 1.1)
    
    # Подсвечиваем лучшую модель
    for i, (bar, model) in enumerate(zip(bars1, models)):
        if model == best_model:
            bar.set_color('gold')
            bar.set_edgecolor('orange')
            bar.set_linewidth(3)
    
    # Добавляем значения на столбцы
    for i, v in enumerate(accuracies):
        axes[0,0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 2. Количество параметров
    params = [resnet_results['model_info']['total_parameters'] / 1e6, 
              efficientnet_results['model_info']['total_parameters'] / 1e6]
    bars2 = axes[0,1].bar(models, params, color=colors)
    axes[0,1].set_title('Number of Parameters (millions)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Parameters (M)')
    
    for i, v in enumerate(params):
        axes[0,1].text(i, v + 0.1, f'{v:.1f}M', ha='center', fontweight='bold')
    
    # 3. Размер модели
    sizes = [resnet_results['model_info']['model_size_mb'], 
             efficientnet_results['model_info']['model_size_mb']]
    bars3 = axes[1,0].bar(models, sizes, color=colors)
    axes[1,0].set_title('Model Size', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Size (MB)')
    
    for i, v in enumerate(sizes):
        axes[1,0].text(i, v + 0.5, f'{v:.1f}MB', ha='center', fontweight='bold')
    
    # 4. Время обучения
    times = [resnet_results['training_time_minutes'], 
             efficientnet_results['training_time_minutes']]
    bars4 = axes[1,1].bar(models, times, color=colors)
    axes[1,1].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Time (minutes)')
    
    for i, v in enumerate(times):
        axes[1,1].text(i, v + 0.5, f'{v:.1f}min', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ДАННЫЕ График сравнения сохранен как 'model_comparison.png'")

def export_best_model(best_model, best_dir):
    """Экспортирует лучшую модель в ONNX"""
    print(f"\nЭКСПОРТ Экспорт лучшей модели {best_model} в ONNX...")
    
    # Находим файл модели
    model_file = best_dir / "best_model.pth"
    if not model_file.exists():
        print(f"ОШИБКА Файл модели не найден: {model_file}")
        return False
    
    # Экспортируем в ONNX
    try:
        # Устанавливаем переменные окружения для отключения эмодзи в PyTorch
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['TORCH_ONNX_DISABLE_EMOJI'] = '1'
        
        result = subprocess.run([
            sys.executable, "export_model.py",
            "--model_path", str(model_file),
            "--output_path", "models/best_model.onnx"
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore', env=env)
        
        if result.returncode == 0:
            print("УСПЕХ Модель успешно экспортирована в ONNX!")
            print("Файл: models/best_model.onnx")
            return True
        else:
            print(f"ОШИБКА Ошибка при экспорте: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ОШИБКА Исключение при экспорте: {e}")
        return False

def main():
    """Основная функция"""
    print("Сравнение и анализ обученных моделей классификации фруктов")
    print("=" * 70)
    print("Информация: Скрипт сначала ищет уже обученные модели")
    print("Информация: Если модели не найдены, запускается обучение")
    print("=" * 70)
    
    # Список моделей для проверки/обучения
    models_to_train = [
        ("resnet18", "ResNet18"),
        ("efficientnet_b0", "EfficientNet-B0")
    ]
    
    training_results = {}
    
    # Обучаем каждую модель
    for model_name, display_name in models_to_train:
        success, training_time = train_model(model_name, epochs=20, lr=0.001, batch_size=32)
        training_results[display_name] = {
            'success': success,
            'training_time': training_time
        }
        
        if not success:
            print(f"ПРЕДУПРЕЖДЕНИЕ Обучение {display_name} не удалось, продолжаем...")
    
    # Сравниваем модели
    comparison_result = compare_models()
    
    if comparison_result:
        best_model, best_dir, best_results = comparison_result
        
        # Экспортируем лучшую модель
        export_success = export_best_model(best_model, best_dir)
        
        # Сохраняем информацию о лучшей модели для веб-приложения
        best_model_info = {
            "best_model_name": best_model,
            "best_model_dir": str(best_dir),
            "best_model_results": best_results,
            "onnx_path": "models/best_model.onnx",
            "export_success": export_success
        }
        
        best_info_path = Path("models") / "best_model_info.json"
        best_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_info_path, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)
        print(f"ДАННЫЕ Информация о лучшей модели сохранена: {best_info_path}")
        
        if export_success:
            print(f"\nУСПЕХ Процесс завершен успешно!")
            print(f"ПОБЕДИТЕЛЬ Лучшая модель: {best_model}")
            print(f"ДАННЫЕ Test Accuracy: {best_results['test_accuracy']:.4f}")
            print(f"ONNX модель: models/best_model.onnx")
            print(f"ВЕБ Готово для веб-приложения!")
        else:
            print(f"\nПРЕДУПРЕЖДЕНИЕ Обучение завершено, но экспорт в ONNX не удался")
    else:
        print(f"\nОШИБКА Не удалось сравнить модели")

if __name__ == "__main__":
    main()
