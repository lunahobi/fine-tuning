#!/usr/bin/env python3
"""
Полный pipeline для обучения моделей и создания веб-приложения
Альтернативная версия с дополнительными возможностями
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
import webbrowser

def run_command(cmd, description, timeout=300, log_file=None):
    print(f"\n🚀 {description}")
    print("=" * 60)
    print(f"Команда: {' '.join(cmd)}")
    
    import subprocess, time, sys
    start_time = time.time()
    if log_file is None:
        # имя файла по описанию
        safe = "".join(c for c in description if c.isalnum() or c in ("_", "-"))
        log_file = Path("logs") / f"{safe}_{int(start_time)}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w", encoding="utf-8") as lf:
        try:
            # Устанавливаем переменные окружения для правильной кодировки
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=".",
                env=env,
            )
            lines = []
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    lf.write(line)
                    lf.flush()
                    lines.append(line)
                # таймаут вручную
                if (time.time() - start_time) > timeout and proc.poll() is None:
                    proc.kill()
                    print(f"⏰ {description} - ТАЙМАУТ ({timeout}с)")
                    return False, f"Timeout. See {log_file}"
            rc = proc.returncode
            duration = time.time() - start_time
            if rc == 0:
                print(f"✅ {description} - УСПЕШНО ({duration:.1f}с)")
                return True, f"See {log_file}"
            else:
                print(f"❌ {description} - ОШИБКА ({duration:.1f}с). Лог: {log_file}")
                return False, f"Return code {rc}. See {log_file}"
        except Exception as e:
            print(f"❌ Исключение: {e}")
            return False, str(e)

def check_requirements():
    """Проверяет наличие необходимых зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'numpy', 'pandas', 
        'sklearn', 'matplotlib', 'seaborn', 'PIL', 
        'onnx', 'onnxruntime', 'gradio', 'tensorboard'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("Установите их командой: pip install -r requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def check_data():
    """Проверяет наличие данных"""
    print("\n📁 Проверка данных...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Папка data/raw не существует")
        print("💡 Создайте папки data/raw/apple, data/raw/kiwi, data/raw/mandarin")
        print("💡 И поместите туда изображения фруктов")
        return False
    
    # Проверяем наличие изображений
    total_images = 0
    for class_name in ['apple', 'kiwi', 'mandarin']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.webp")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            print(f"📸 {class_name}: {len(images)} изображений")
            total_images += len(images)
        else:
            print(f"❌ Папка {class_name} не существует")
    
    if total_images == 0:
        print("❌ Нет изображений для обучения")
        return False
    
    print(f"✅ Найдено {total_images} изображений")
    return True

def save_pipeline_log(results):
    """Сохраняет лог выполнения pipeline"""
    log_file = Path("pipeline_log.json")
    
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "total_time": sum(r.get("duration", 0) for r in results.values()),
        "success": all(r.get("success", False) for r in results.values())
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Лог сохранен: {log_file}")

def main():
    """Основная функция pipeline"""
    print("🍎🥝🍊 ПОЛНЫЙ PIPELINE ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    # Проверяем, что мы в правильной директории (проверяем наличие ключевых файлов)
    required_files = ["train.py", "data_preprocessing.py", "train_both_models.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Ошибка: Запустите скрипт из корневой директории проекта")
        print(f"   Не найдены файлы: {', '.join(missing_files)}")
        return
    
    # Шаг 0: Проверка зависимостей
    print("\n📋 ШАГ 0: Проверка зависимостей")
    if not check_requirements():
        print("❌ Не все зависимости установлены")
        return
    
    # Шаг 0.5: Проверка данных
    print("\n📁 ШАГ 0.5: Проверка данных")
    if not check_data():
        print("❌ Данные не готовы")
        return
    
    # Шаг 1: Подготовка данных
    print("\n📊 ШАГ 1: Подготовка данных")
    success, output = run_command([sys.executable, "data_preprocessing.py"], "Подготовка и разделение данных")
    results["data_preprocessing"] = {"success": success, "output": output}
    
    if not success:
        print("❌ Не удалось подготовить данные")
        save_pipeline_log(results)
        return
    
    # Шаг 2: Обучение ResNet18 (train.py)
    print("\n🏗️ ШАГ 2: Обучение ResNet18")
    success, output = run_command([
        sys.executable, "train.py",
        "--model", "resnet18",
        "--epochs", "20",
        "--lr", "0.001",
        "--freeze_epochs", "5",
        "--head_lr", "0.001",
        "--backbone_lr", "0.0002",
        "--batch_size", "32"
    ], "Обучение ResNet18", timeout=1800)
    results["resnet18_training"] = {"success": success, "output": output}
    
    if not success:
        print("⚠️ Обучение ResNet18 не удалось, продолжаем...")
    
    # Шаг 3: Обучение EfficientNet-B0 (train.py)
    print("\n🏗️ ШАГ 3: Обучение EfficientNet-B0")
    success, output = run_command([
        sys.executable, "train.py",
        "--model", "efficientnet_b0",
        "--epochs", "20",
        "--lr", "0.001",
        "--freeze_epochs", "5",
        "--head_lr", "0.001",
        "--backbone_lr", "0.0001",
        "--batch_size", "32"
    ], "Обучение EfficientNet-B0", timeout=1800)  # 30 минут таймаут
    results["efficientnet_training"] = {"success": success, "output": output}
    
    if not success:
        print("⚠️ Обучение EfficientNet-B0 не удалось, продолжаем...")
    
    # Проверяем, что хотя бы одна модель обучена перед сравнением
    experiments_dir = Path("experiments")
    has_models = False
    if experiments_dir.exists():
        model_dirs = [d for d in experiments_dir.iterdir() 
                     if d.is_dir() and (("resnet18" in d.name.lower()) or ("efficientnet" in d.name.lower()))]
        has_models = any((d / "results.json").exists() and (d / "best_model.pth").exists() for d in model_dirs)
    
    if not has_models:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Нет обученных моделей для сравнения!")
        print("   Обучите хотя бы одну модель перед продолжением")
        save_pipeline_log(results)
        return
    
    # Шаг 4: Сравнение и выбор лучшей модели
    print("\n📊 ШАГ 4: Сравнение моделей")
    success, output = run_command([sys.executable, "train_both_models.py"], "Сравнение и выбор лучшей модели", timeout=600)
    results["model_comparison"] = {"success": success, "output": output}
    
    if not success:
        print("⚠️ Сравнение моделей не удалось, но продолжаем...")
        print("💡 Запустите вручную: python train_both_models.py")
    
    # Шаг 4.5: Дополнительный экспорт обеих моделей в ONNX (опционально, для резерва)
    # Примечание: train_both_models.py уже экспортирует лучшую модель в models/best_model.onnx
    # Этот шаг экспортирует обе модели отдельно, что может быть полезно
    print("\n📦 ШАГ 4.5: Дополнительный экспорт моделей в ONNX (опционально)")
    print("   Примечание: Лучшая модель уже экспортирована в models/best_model.onnx")
    print("   Этот шаг экспортирует обе модели отдельно для резерва")
    
    # Находим последние эксперименты для каждой модели
    experiments_dir = Path("experiments")
    resnet_model_path = None
    efficientnet_model_path = None
    
    if experiments_dir.exists():
        # Ищем последний эксперимент ResNet18
        resnet_dirs = [d for d in experiments_dir.iterdir() 
                      if d.is_dir() and "resnet18" in d.name.lower()]
        if resnet_dirs:
            resnet_latest = max(resnet_dirs, key=lambda x: x.stat().st_mtime)
            resnet_model_file = resnet_latest / "best_model.pth"
            if resnet_model_file.exists():
                resnet_model_path = resnet_model_file
        
        # Ищем последний эксперимент EfficientNet
        efficientnet_dirs = [d for d in experiments_dir.iterdir() 
                            if d.is_dir() and "efficientnet" in d.name.lower()]
        if efficientnet_dirs:
            efficientnet_latest = max(efficientnet_dirs, key=lambda x: x.stat().st_mtime)
            efficientnet_model_file = efficientnet_latest / "best_model.pth"
            if efficientnet_model_file.exists():
                efficientnet_model_path = efficientnet_model_file
    
    # Экспортируем ResNet18 в ONNX (если найден)
    if resnet_model_path:
        print(f"\n📦 Экспорт ResNet18 (резерв): {resnet_model_path.name}")
        success, output = run_command([
            sys.executable, "export_model.py",
            "--model_path", str(resnet_model_path),
            "--output_path", "models/resnet18_final.onnx"
        ], "Экспорт ResNet18 в ONNX", timeout=300)
        results["resnet18_export"] = {"success": success, "output": output}
    else:
        print("⚠️ ResNet18 модель не найдена для экспорта")
        results["resnet18_export"] = {"success": False, "output": "Model not found"}
    
    # Экспортируем EfficientNet-B0 в ONNX (если найден)
    if efficientnet_model_path:
        print(f"\n📦 Экспорт EfficientNet-B0 (резерв): {efficientnet_model_path.name}")
        success, output = run_command([
            sys.executable, "export_model.py",
            "--model_path", str(efficientnet_model_path),
            "--output_path", "models/efficientnet_b0_final.onnx"
        ], "Экспорт EfficientNet-B0 в ONNX", timeout=300)
        results["efficientnet_export"] = {"success": success, "output": output}
    else:
        print("⚠️ EfficientNet-B0 модель не найдена для экспорта")
        results["efficientnet_export"] = {"success": False, "output": "Model not found"}
    
    # Шаг 5: Запуск Gradio приложения с лучшей моделью ONNX
    print("\n🚀 ШАГ 5: Запуск Gradio-приложения best_model_classifier.py")
    try:
        process = subprocess.Popen([
            sys.executable, "app/best_model_classifier.py",
            "--port", "8080"
        ])
        print("✅ Приложение запущено!")
        print("🌐 Откройте браузер: http://localhost:8080")
        try:
            webbrowser.open("http://localhost:8080")
        except:
            pass
        results["web_app"] = {"success": True, "type": "gradio", "url": "http://localhost:8080"}
    except Exception as e:
        print(f"❌ Ошибка при запуске приложения: {e}")
        results["web_app"] = {"success": False, "error": str(e)}
    
    # Финальная статистика
    total_time = time.time() - start_time
    successful_steps = sum(1 for r in results.values() if r.get("success", False))
    total_steps = len(results)
    
    print(f"\n🎉 PIPELINE ЗАВЕРШЕН!")
    print(f"⏱️ Общее время: {total_time/60:.1f} минут")
    print(f"✅ Успешных шагов: {successful_steps}/{total_steps}")
    
    # Сохраняем лог
    for key, value in results.items():
        value["duration"] = total_time / total_steps  # Примерное время на шаг
    
    save_pipeline_log(results)
    
    if successful_steps == total_steps:
        print("🏆 Все шаги выполнены успешно!")
    else:
        print("⚠️ Некоторые шаги не удались, проверьте лог")

if __name__ == "__main__":
    main()
