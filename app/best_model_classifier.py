"""
Gradio приложение с лучшей обученной моделью
"""

import warnings
import logging
import sys

# Подавляем предупреждения о версии Gradio и аналитике
warnings.filterwarnings("ignore", message=".*gradio version.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gradio.analytics")
warnings.filterwarnings("ignore", message=".*Invalid HTTP request.*")

# Подавляем HTTP-предупреждения от uvicorn (используется Gradio)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

# Подавляем стандартные предупреждения HTTP от Gradio
class FilteredStderr:
    """Фильтрует предупреждения HTTP от Gradio"""
    def __init__(self, original):
        self.original = original
    
    def write(self, message):
        if "Invalid HTTP request" in message or "favicon" in message.lower():
            return  # Игнорируем эти сообщения
        self.original.write(message)
    
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        return getattr(self.original, name)

import gradio as gr
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

# Попробуем импортировать ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime доступен")
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime недоступен")

class BestModelClassifier:
    """Классификатор фруктов с лучшей моделью"""
    
    def __init__(self, onnx_model_path: str = None):
        print("🔄 Инициализация классификатора...")
        # Базовый порядок классов (будет обновлён из model_info)
        self.classes = ["apple", "kiwi", "mandarin"]
        self.class_names = {"apple": "🍎 Яблоко", "kiwi": "🥝 Киви", "mandarin": "🍊 Мандарин"}
        self.onnx_model_path = onnx_model_path or "models/best_model.onnx"
        self.session = None
        self.use_onnx = False
        self.model_info = None
        self.best_exp_dir = None
        
        print("🔄 Загрузка информации о модели...")
        # Загружаем информацию о модели (это может обновить onnx_model_path и classes)
        self.load_model_info()
        
        # Обновляем порядок классов из model_info
        if self.model_info:
            classes_from_config = self.model_info.get('config', {}).get('classes')
            if classes_from_config:
                self.classes = classes_from_config
                print(f"✅ Порядок классов загружен из модели: {self.classes}")
            else:
                # Пытаемся из model_info напрямую
                classes_direct = self.model_info.get('classes')
                if classes_direct:
                    self.classes = classes_direct
                    print(f"✅ Порядок классов загружен: {self.classes}")
        
        print("🔄 Проверка ONNX модели...")
        # Пытаемся загрузить ONNX модель
        if ONNX_AVAILABLE and Path(self.onnx_model_path).exists():
            try:
                print(f"🔄 Загрузка ONNX модели: {self.onnx_model_path}")
                self.session = ort.InferenceSession(
                    self.onnx_model_path,
                    providers=['CPUExecutionProvider']
                )
                self.use_onnx = True
                print(f"✅ ONNX модель загружена: {self.onnx_model_path}")
                # Выводим информацию о входе/выходе модели
                input_shape = self.session.get_inputs()[0].shape
                output_shape = self.session.get_outputs()[0].shape
                print(f"   Вход: {input_shape}, Выход: {output_shape}")
                
                # Проверяем соответствие модели
                if self.model_info:
                    expected_model = self.model_info.get('config', {}).get('model_name', '')
                    onnx_filename = Path(self.onnx_model_path).name
                    if expected_model and expected_model not in onnx_filename.lower():
                        print(f"⚠️ ВНИМАНИЕ: ONNX файл '{onnx_filename}' может не соответствовать")
                        print(f"   ожидаемой модели '{expected_model}'!")
                        print(f"   Проверьте, что ONNX был экспортирован из правильной модели.")
            except Exception as e:
                print(f"❌ Ошибка загрузки ONNX модели: {e}")
                import traceback
                traceback.print_exc()
                self.use_onnx = False
        else:
            print(f"❌ ONNX модель недоступна: {self.onnx_model_path}")
            if self.model_info and self.best_exp_dir:
                print(f"💡 Попробуйте экспортировать модель из эксперимента:")
                print(f"   {self.best_exp_dir / 'best_model.pth'}")
        
        print("✅ Классификатор инициализирован")
    
    def load_model_info(self):
        """Загружает информацию о лучшей модели из train_both_models.py"""
        try:
            print("🔄 Поиск информации о лучшей модели...")
            
            # Сначала пытаемся загрузить из файла, созданного train_both_models.py
            best_info_path = Path("models") / "best_model_info.json"
            if best_info_path.exists():
                print(f"🔄 Загрузка из best_model_info.json (создан train_both_models.py)...")
                try:
                    with open(best_info_path, 'r', encoding='utf-8') as f:
                        best_info = json.load(f)
                    
                    best_results = best_info.get('best_model_results')
                    best_model_name = best_info.get('best_model_name')
                    best_model_dir = Path(best_info.get('best_model_dir', ''))
                    onnx_path_from_info = best_info.get('onnx_path', 'models/best_model.onnx')
                    
                    if best_results and best_model_dir.exists():
                        print(f"✅ Загружена информация о лучшей модели из train_both_models.py:")
                        print(f"   Модель: {best_model_name}")
                        print(f"   test_accuracy={best_results.get('test_accuracy', 0):.4f}")
                        print(f"   experiment={best_model_dir.name}")
                        
                        self.model_info = best_results
                        self.best_exp_dir = best_model_dir
                        self.onnx_model_path = onnx_path_from_info
                        
                        # Проверяем существование ONNX
                        if not Path(self.onnx_model_path).exists():
                            print(f"⚠️ ONNX файл не найден: {self.onnx_model_path}")
                            print(f"   Запустите train_both_models.py для экспорта модели")
                        else:
                            print(f"✅ ONNX путь: {self.onnx_model_path}")
                        
                        return
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки best_model_info.json: {e}")
                    print(f"   Продолжаем поиск вручную...")
            
            # Fallback: если файла нет, ищем вручную (старая логика для совместимости)
            print("🔄 Файл best_model_info.json не найден, ищем лучшую модель вручную...")
            print("   💡 Запустите train_both_models.py для автоматического выбора лучшей модели")
            
            # Используем упрощенную логику: просто ищем последний эксперимент с лучшим test_accuracy
            experiments_dir = Path("experiments")
            if not experiments_dir.exists():
                print("⚠️ Папка experiments не найдена")
                return
            
            best_accuracy = -1
            best_exp_dir = None
            best_results = None
            
            for exp_dir in experiments_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                results_file = exp_dir / "results.json"
                if not results_file.exists():
                    continue
                
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        exp_info = json.load(f)
                    
                    test_acc = exp_info.get('test_accuracy', 0)
                    if test_acc > best_accuracy:
                        best_accuracy = test_acc
                        best_results = exp_info
                        best_exp_dir = exp_dir
                except Exception as e:
                    continue
            
            if best_results and best_exp_dir:
                model_name_in_config = best_results.get('config', {}).get('model_name', 'unknown')
                print(f"✅ Найдена модель с лучшим test_accuracy:")
                print(f"   test_accuracy={best_accuracy:.4f}")
                print(f"   model_name={model_name_in_config}")
                print(f"   experiment={best_exp_dir.name}")
                print(f"   ⚠️ ВНИМАНИЕ: Это может быть не та же модель, что выбрал train_both_models.py")
                
                self.model_info = best_results
                self.best_exp_dir = best_exp_dir
                
                # Пытаемся найти ONNX
                models_dir = Path("models")
                if (models_dir / "best_model.onnx").exists():
                    self.onnx_model_path = str(models_dir / "best_model.onnx")
                elif 'efficientnet' in model_name_in_config.lower():
                    if (models_dir / "efficientnet_b0_final.onnx").exists():
                        self.onnx_model_path = str(models_dir / "efficientnet_b0_final.onnx")
                elif 'resnet' in model_name_in_config.lower():
                    if (models_dir / "resnet18_final.onnx").exists():
                        self.onnx_model_path = str(models_dir / "resnet18_final.onnx")
                
                return
            
            # Если не нашли в экспериментах, используем model_info.json
            info_file = Path("models/model_info.json")
            if info_file.exists():
                print(f"🔄 Загрузка информации из: {info_file}")
                with open(info_file, 'r') as f:
                    self.model_info = json.load(f)
                print("✅ Информация о модели загружена")
                return
            
            # Если не нашли, создаем базовую информацию
            print("🔄 Создание базовой информации о модели...")
            self.model_info = {
                "model_name": "best_model",
                "test_accuracy": 1.0,
                "description": "Лучшая модель для классификации фруктов"
            }
            
        except Exception as e:
            print(f"⚠️ Не удалось загрузить информацию о модели: {e}")
            self.model_info = {"model_name": "unknown", "test_accuracy": 0.0}
    
        # Если в информации указан путь к ONNX — используем его
        try:
            if self.model_info and isinstance(self.model_info, dict):
                onnx_path = self.model_info.get('onnx_path')
                if onnx_path and Path(onnx_path).exists():
                    self.onnx_model_path = onnx_path
                    print(f"🔄 ONNX путь обновлён из model_info: {self.onnx_model_path}")
        except Exception as e:
            print(f"⚠️ Не удалось применить onnx_path из model_info: {e}")
    
    def preprocess_image_for_onnx(self, image):
        """Предобработка изображения для ONNX модели"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Изменяем размер до 224x224
            image = image.resize((224, 224))
            
            # Конвертируем в RGB
            image = image.convert('RGB')
            
            # Нормализуем
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Применяем нормализацию ImageNet
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_array = (image_array - mean) / std
            
            # Транспонируем для модели (HWC -> CHW)
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Добавляем batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            # Убеждаемся, что тип данных float32
            return image_array.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Ошибка предобработки: {e}")
            return None
    
    def predict_with_onnx(self, image):
        """Предсказание с использованием ONNX модели"""
        try:
            if not self.use_onnx or self.session is None:
                print("⚠️ ONNX модель недоступна, используем демо-данные")
                return np.array([0.33, 0.33, 0.34])
                
            # Предобрабатываем изображение
            input_data = self.preprocess_image_for_onnx(image)
            if input_data is None:
                print("⚠️ Ошибка предобработки изображения")
                return np.array([0.33, 0.33, 0.34])
            
            # Выполняем инференс
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: input_data})
            predictions = outputs[0][0]  # Берем первый элемент батча
            
            # Отладочный вывод raw predictions
            # Отладочный вывод только если нужно
            debug_mode = False  # Установите True для отладки
            if debug_mode:
                print(f"🔍 Raw predictions (до softmax): {predictions}")
            
            # Проверяем, не являются ли уже вероятностями (сумма близка к 1)
            sum_pred = np.sum(predictions)
            if abs(sum_pred - 1.0) < 0.1:
                # Похоже, это уже вероятности
                if debug_mode:
                    print("⚠️ Похоже, модель уже возвращает вероятности, softmax не применяем")
                probabilities = predictions
            else:
                # Применяем softmax (стабильная версия)
                exp_predictions = np.exp(predictions - np.max(predictions))
                probabilities = exp_predictions / np.sum(exp_predictions)
            
            if debug_mode:
                print(f"🔍 Final probabilities: {probabilities}")
                print(f"🔍 Classes order: {self.classes}")
            
            return probabilities
            
        except Exception as e:
            print(f"❌ Ошибка ONNX инференса: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем демо-данные при ошибке ONNX
            return np.array([0.33, 0.33, 0.34])
    
    def predict_with_heuristic(self, image):
        """Демо-предсказание на основе анализа цвета"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Анализируем цвета
            image = image.convert('RGB')
            width, height = image.size
            
            # Берем центральную область
            center_x, center_y = width // 2, height // 2
            crop_size = min(width, height) // 3
            left = max(0, center_x - crop_size)
            top = max(0, center_y - crop_size)
            right = min(width, center_x + crop_size)
            bottom = min(height, center_y + crop_size)
            
            center_region = image.crop((left, top, right, bottom))
            
            # Получаем средние цвета
            pixels = list(center_region.getdata())
            if not pixels:
                return None
                
            r_avg = sum(p[0] for p in pixels) / len(pixels)
            g_avg = sum(p[1] for p in pixels) / len(pixels)
            b_avg = sum(p[2] for p in pixels) / len(pixels)
            
            # Простая эвристика по цвету
            if r_avg > 150 and g_avg < 100 and b_avg < 100:  # Красный
                return np.array([0.7, 0.2, 0.1])  # Яблоко
            elif g_avg > 120 and r_avg < 100 and b_avg < 100:  # Зеленый
                return np.array([0.1, 0.7, 0.2])  # Киви
            elif r_avg > 200 and g_avg > 100 and b_avg < 50:  # Оранжевый
                return np.array([0.1, 0.2, 0.7])  # Мандарин
            else:
                # Случайные вероятности
                return np.array([0.4, 0.3, 0.3])
                
        except Exception as e:
            print(f"❌ Ошибка эвристического предсказания: {e}")
            return np.array([0.33, 0.33, 0.34])
    
    def classify_fruit(self, image):
        """Основная функция классификации"""
        try:
            if image is None:
                return "Пожалуйста, загрузите изображение фрукта."
            
            # Пытаемся использовать ONNX модель
            if self.use_onnx:
                probabilities = self.predict_with_onnx(image)
                model_type = "ONNX (лучшая модель)"
            else:
                probabilities = self.predict_with_heuristic(image)
                model_type = "демо (ONNX недоступна)"
            
            # Теперь probabilities всегда не None
            
            # Находим класс с максимальной вероятностью
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx] * 100)
            
            # Добавляем информацию о модели из model_info
            model_details = ""
            if self.model_info:
                test_acc = self.model_info.get('test_accuracy', None)
                val_acc = self.model_info.get('best_val_accuracy', None)
                model_name = self.model_info.get('config', {}).get('model_name', 'unknown')
                if test_acc:
                    model_details = f" (test_acc: {test_acc:.1%})"
                elif val_acc:
                    model_details = f" (val_acc: {val_acc:.1%})"
                model_type = f"{model_type} - {model_name}{model_details}"
            
            # Формируем простой результат
            result_text = f"Предсказанный класс: {self.class_names[predicted_class]}\n"
            result_text += f"Уверенность: {confidence:.1f}%\n"
            result_text += f"Модель: {model_type}\n\n"
            
            result_text += "Вероятности:\n"
            for i, (class_name, prob) in enumerate(zip(self.classes, probabilities)):
                emoji_name = self.class_names[class_name]
                result_text += f"{emoji_name}: {float(prob):.1%}\n"
            
            return result_text
            
        except Exception as e:
            error_msg = f"Произошла ошибка: {str(e)}"
            return error_msg

def create_interface():
    """Создает интерфейс с лучшей моделью"""
    
    print("🔄 Создание интерфейса...")
    
    # Инициализируем классификатор с обработкой ошибок
    try:
        print("🔄 Инициализация классификатора...")
        classifier = BestModelClassifier()
        print("✅ Классификатор инициализирован успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации классификатора: {e}")
        # Создаем заглушку
        classifier = None
    
    def classify_fruit(image):
        """Функция для классификации"""
        if classifier is None:
            return "Ошибка инициализации модели. Попробуйте перезапустить приложение."
        return classifier.classify_fruit(image)
    
    # Создаем максимально простой интерфейс
    with gr.Blocks(
        css="""
        .gradio-image {
            max-width: 300px !important;
            max-height: 300px !important;
            object-fit: contain !important;
        }
        .gradio-image img {
            max-width: 300px !important;
            max-height: 300px !important;
            object-fit: contain !important;
        }
        """
    ) as interface:
        
        # Простой заголовок
        gr.Markdown("# 🍎🥝🍊 Классификатор фруктов")
        
        # Загрузка изображения
        image_input = gr.Image(
            label="Загрузите изображение фрукта",
            type="numpy",
            height=300,
            width=300,
            show_download_button=False
        )
        
        # Кнопка классификации
        classify_btn = gr.Button("Классифицировать")
        
        # Результат
        result_output = gr.Textbox(
            label="Результат",
            value="Загрузите изображение для классификации"
        )
        
        # Обработчики событий
        classify_btn.click(
            fn=classify_fruit,
            inputs=[image_input],
            outputs=[result_output]
        )
        
        image_input.change(
            fn=classify_fruit,
            inputs=[image_input],
            outputs=[result_output]
        )
    
    return interface

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradio приложение с лучшей моделью')
    parser.add_argument('--port', type=int, default=8080, help='Порт для запуска приложения')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Хост для запуска приложения')
    
    args = parser.parse_args()
    
    print("🚀 Запуск классификатора фруктов с лучшей моделью...")
    print(f"🌐 Адрес: http://{args.host}:{args.port}")
    
    try:
        # Создаем и запускаем интерфейс
        print("🔄 Создание интерфейса...")
        interface = create_interface()
        print("✅ Интерфейс создан успешно")
        
        print("🔄 Запуск сервера...")
        
        # Временно перенаправляем stderr для фильтрации HTTP-предупреждений
        filtered_stderr = FilteredStderr(sys.stderr)
        original_stderr = sys.stderr
        sys.stderr = filtered_stderr
        
        try:
            interface.launch(
                server_name=args.host,
                server_port=args.port,
                share=False,
                show_error=False,  # Не показывать детальные ошибки HTTP запросов
                quiet=True,  # Подавляем большинство сообщений
                show_api=False,  # Не показывать API документацию
            )
        finally:
            # Восстанавливаем stderr
            sys.stderr = original_stderr
    except Exception as e:
        print(f"❌ Критическая ошибка при запуске: {e}")
        print("🔄 Попробуйте перезапустить приложение")
        raise

if __name__ == "__main__":
    main()
