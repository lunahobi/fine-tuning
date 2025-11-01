"""
Улучшенный скрипт обучения с сохранением графиков на каждой эпохе
"""

import argparse
import json
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
# Добавляем корень проекта в sys.path
sys.path.append(str(Path(__file__).parent))

# Исправление кодировки для Windows консоли
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from training.config import Config
from training.utils import set_random_seeds, get_device, print_system_info
from data_preprocessing import split_dataset, validate_images

class FruitDataset(torch.utils.data.Dataset):
    """Датасет для классификации фруктов"""
    
    def __init__(self, data_dir: Path, classes: list, transform=None):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        
        # Собираем все файлы
        for class_idx, class_name in enumerate(classes):
            class_dir = data_dir / class_name
            if class_dir.exists():
                files = []
                # поддерживаем популярные расширения
                for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png"):
                    files.extend(class_dir.glob(ext))
                for img_file in files:
                    self.samples.append((img_file, class_idx))
        
        print(f"Найдено образцов в {data_dir.name}: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем случайное изображение в случае ошибки
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label

def get_transforms():
    """Получает трансформации для обучения и валидации"""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(config):
    """Создает модель"""
    print(f"Создание модели: {config.model.model_name}")
    
    model = timm.create_model(
        config.model.model_name,
        pretrained=config.model.pretrained,
        num_classes=config.data.num_classes
    )
    
    print(f"Модель создана: {config.model.model_name}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def _is_head_param(name: str) -> bool:
    """Heuristic to detect classifier head parameters across timm models.
    Works for resnet (fc.*), efficientnet (classifier.*), and many timm heads (head.*).
    """
    if 'classifier' in name:
        return True
    if name.startswith('fc.'):
        return True
    if name.startswith('head.'):
        return True
    # some models use last_linear
    if name.startswith('last_linear.'):
        return True
    return False


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze all backbone params, keep classifier head trainable."""
    for name, p in model.named_parameters():
        p.requires_grad = _is_head_param(name)


def unfreeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

def build_optimizer(model, opt_name: str, head_lr: float, backbone_lr: float):
    """Create optimizer with separate LR for head/backbone."""
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if _is_head_param(name) else backbone_params).append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})

    name = opt_name.lower()
    if name == 'adamw':
        return optim.AdamW(param_groups)
    if name == 'adam':
        return optim.Adam(param_groups)
    if name == 'sgd':
        return optim.SGD(param_groups, momentum=0.9)
    # default
    return optim.AdamW(param_groups)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Обучает модель одну эпоху"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Создаем прогресс-бар для батчей с явным позиционированием
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Эпоха {epoch} - Обучение", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                leave=False,
                position=0)
    
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Обновляем прогресс-бар с текущими метриками
        current_acc = 100. * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg_Loss': f'{avg_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'Correct': f'{correct}/{total}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Закрываем прогресс-бар перед выводом результатов
    pbar.close()
    tqdm.write(f"\nЭпоха {epoch} завершена:")
    tqdm.write(f"   Средняя потеря: {epoch_loss:.4f}")
    tqdm.write(f"   Точность: {epoch_acc:.2f}%")
    tqdm.write(f"   Правильных предсказаний: {correct}/{total}")
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Валидирует модель"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Создаем прогресс-бар для валидации с явным позиционированием
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                desc="Валидация", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                leave=False,
                position=0)
    
    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Обновляем прогресс-бар с текущими метриками
            current_acc = 100. * correct / total
            avg_loss = val_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Correct': f'{correct}/{total}'
            })
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    # Закрываем прогресс-бар перед выводом результатов
    pbar.close()
    tqdm.write(f"\nВалидация завершена:")
    tqdm.write(f"   Валидационная потеря: {val_loss:.4f}")
    tqdm.write(f"   Валидационная точность: {val_acc:.2f}%")
    tqdm.write(f"   Правильных предсказаний: {correct}/{total}")
    
    return val_loss, val_acc

def save_training_plots(train_losses, val_losses, train_accs, val_accs, epoch, save_dir):
    """Сохраняет графики обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, epoch + 1)
    
    # График потерь
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title(f'Training and Validation Loss (Epoch {epoch})', fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # График точности
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'Training and Validation Accuracy (Epoch {epoch})', fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'training_epoch_{epoch:03d}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model, test_loader, device, classes):
    """Тестирует модель"""
    print(f"\nНачало тестирования...")
    print(f"   Количество батчей тестирования: {len(test_loader)}")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Создаем прогресс-бар для тестирования
    pbar = tqdm(test_loader, desc="Тестирование", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Обновляем прогресс-бар
            current_acc = accuracy_score(all_targets, all_predictions) * 100
            pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    # Вычисляем метрики
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, target_names=classes)
    
    print(f"\nТестирование завершено:")
    print(f"   Обработано: {len(all_targets)} изображений")
    print(f"   Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, cm, report

def plot_confusion_matrix(cm, classes, save_path):
    """Создает и сохраняет confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(config):
    """Основная функция обучения"""
    print("="*80)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    print("="*80)
    print(f"Конфигурация:")
    print(f"   Модель: {config.model.model_name}")
    print(f"   Эпохи: {config.model.num_epochs}")
    print(f"   Learning Rate: {config.model.learning_rate}")
    print(f"   Оптимизатор: {config.model.optimizer}")
    print(f"   Batch Size: {config.data.batch_size}")
    print(f"   Классы: {config.data.classes}")
    print(f"   Эксперимент: {config.experiment.experiment_name}")
    print(f"   Seed: {config.experiment.random_seed}")
    print("="*80)
    
    # Устанавливаем воспроизводимость
    set_random_seeds(config.experiment.random_seed)
    
    # Получаем устройство
    device = get_device(config.model.device)
    print(f"Используемое устройство: {device}")
    
    # Создаем директории
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.model.model_name}_{config.experiment.experiment_name}_{timestamp}"
    results_dir = config.data.experiments_dir / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Результаты будут сохранены в: {results_dir}")
    print("="*80)
    
    # Получаем трансформации
    print("Загрузка трансформаций данных...")
    train_transform, val_transform = get_transforms()
    print("Трансформации загружены")
    
    # Создаем датасеты
    print("\nЗагрузка датасетов...")
    print("   Загрузка обучающего датасета...")
    train_dataset = FruitDataset(
        config.data.processed_data_dir / 'train',
        config.data.classes,
        train_transform
    )
    print("   Загрузка валидационного датасета...")
    val_dataset = FruitDataset(
        config.data.processed_data_dir / 'val',
        config.data.classes,
        val_transform
    )
    print("   Загрузка тестового датасета...")
    test_dataset = FruitDataset(
        config.data.processed_data_dir / 'test',
        config.data.classes,
        val_transform
    )
    
    print(f"\nРазмеры датасетов:")
    print(f"   Обучающий: {len(train_dataset)} изображений")
    print(f"   Валидационный: {len(val_dataset)} изображений")
    print(f"   Тестовый: {len(test_dataset)} изображений")
    print(f"   Всего: {len(train_dataset) + len(val_dataset) + len(test_dataset)} изображений")
    
    if len(train_dataset) == 0:
        print("Ошибка: Обучающий датасет пуст! Проверьте, что data/processed/train содержит изображения (webp/jpg/png) по классам.")
        sys.exit(1)
    
    # Создаем загрузчики данных
    print("\nСоздание загрузчиков данных...")
    train_loader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, 
        shuffle=True, num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.data.batch_size, 
        shuffle=False, num_workers=config.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.data.batch_size, 
        shuffle=False, num_workers=config.data.num_workers
    )
    print("Загрузчики данных созданы")
    
    # Создаем модель
    print(f"\nСоздание модели {config.model.model_name}...")
    model = create_model(config)
    model = model.to(device)
    print("Модель создана и перемещена на устройство")
    
    # Стратегия: freeze -> unfreeze
    freeze_epochs = getattr(config.model, 'freeze_epochs', 5)
    head_lr = getattr(config.model, 'head_lr', max(1e-3, config.model.learning_rate))
    backbone_lr_unfrozen = getattr(config.model, 'backbone_lr', max(1e-4, config.model.learning_rate / 5))

    print("\nСтратегия обучения:")
    print(f"  Freeze backbone на первые {freeze_epochs} эпох")
    print(f"  Head LR: {head_lr}")
    print(f"  Backbone LR после разморозки: {backbone_lr_unfrozen}")

    # Изначально: заморозить backbone
    freeze_backbone(model)
    optimizer = build_optimizer(model, config.model.optimizer, head_lr=head_lr, backbone_lr=0.0)
    criterion = nn.CrossEntropyLoss()
    print("Оптимизатор и функция потерь настроены")
    
    # TensorBoard
    if config.experiment.use_tensorboard:
        writer = SummaryWriter(log_dir=results_dir / 'tensorboard')
    
    # Обучение
    best_val_acc = 0.0
    best_model_path = results_dir / 'best_model.pth'  # Инициализируем путь к лучшей модели
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    start_time = time.time()
    
    # Создаем общий прогресс-бар для всех эпох
    epoch_pbar = tqdm(range(1, config.model.num_epochs + 1), 
                      desc="Обучение модели", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in epoch_pbar:
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"ЭПОХА {epoch}/{config.model.num_epochs}")
        tqdm.write(f"{'='*60}")
        
        # Обновляем описание прогресс-бара
        epoch_pbar.set_description(f"Эпоха {epoch}/{config.model.num_epochs}")
        
        # Переход к разморозке на первый epoch > freeze_epochs
        if epoch == freeze_epochs + 1:
            tqdm.write("\n==> Размораживаем backbone и пересобираем оптимизатор")
            unfreeze_all(model)
            optimizer = build_optimizer(model, config.model.optimizer, head_lr=head_lr, backbone_lr=backbone_lr_unfrozen)

        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Валидация
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Сохраняем метрики
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        tqdm.write(f"\nИТОГИ ЭПОХИ {epoch}:")
        tqdm.write(f"   Обучение:  Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        tqdm.write(f"   Валидация: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Проверяем улучшение и сохраняем лучшую модель
        is_improvement = val_acc > best_val_acc
        is_first_epoch = epoch == 1
        
        if is_improvement:
            improvement = val_acc - best_val_acc
            tqdm.write(f"   НОВЫЙ РЕКОРД! Улучшение на {improvement:.2f}%")
            best_val_acc = val_acc
        else:
            tqdm.write(f"   Без улучшения (лучший: {best_val_acc:.2f}%)")
        
        # Обновляем прогресс-бар с текущими метриками
        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Acc': f'{val_acc:.2f}%',
            'Best_Val': f'{best_val_acc:.2f}%',
            'Train_Loss': f'{train_loss:.4f}',
            'Val_Loss': f'{val_loss:.4f}'
        })
        
        # TensorBoard логирование
        if config.experiment.use_tensorboard:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # Сохраняем графики на каждой эпохе
        if config.experiment.save_plots_every_epoch:
            save_training_plots(train_losses, val_losses, train_accs, val_accs, epoch, results_dir)
        
        # Сохраняем лучшую модель (по val_accuracy)
        # Сохраняем на первой эпохе или когда val_acc улучшилось
        if is_first_epoch or is_improvement:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'model_name': config.model.model_name,
                    'num_epochs': config.model.num_epochs,
                    'learning_rate': config.model.learning_rate,
                    'optimizer': config.model.optimizer,
                    'device': config.model.device,
                    'classes': config.data.classes,
                    'batch_size': config.data.batch_size,
                    'experiment_name': config.experiment.experiment_name,
                    'random_seed': config.experiment.random_seed
                }
            }, best_model_path)
            if epoch == 1:
                tqdm.write(f"УСПЕХ: Первая модель сохранена! Val Acc: {val_acc:.2f}%")
            else:
                tqdm.write(f"УСПЕХ: Новая лучшая модель сохранена! Val Acc: {val_acc:.2f}%")
        
        # Сохраняем чекпоинт
        if config.model.save_checkpoints and epoch % config.model.checkpoint_freq == 0:
            checkpoint_path = results_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'model_name': config.model.model_name,
                    'num_epochs': config.model.num_epochs,
                    'learning_rate': config.model.learning_rate,
                    'optimizer': config.model.optimizer,
                    'device': config.model.device,
                    'classes': config.data.classes,
                    'batch_size': config.data.batch_size,
                    'experiment_name': config.experiment.experiment_name,
                    'random_seed': config.experiment.random_seed
                }
            }, checkpoint_path)
    
    training_time = time.time() - start_time
    print(f"\nВремя обучения: {training_time/60:.2f} минут")
    
    # Тестирование лучшей модели
    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Лучшая val_accuracy: {best_val_acc:.2f}%")
    print(f"Размер тестового набора: {len(test_dataset)} изображений")
    print(f"Путь к модели: {best_model_path}")
    
    # ВАЖНО: Тестируем модель с лучшей val_accuracy
    # Если test_accuracy сильно ниже val_accuracy - это признак переобучения!
    
    # Проверяем существование файла модели
    if not best_model_path.exists():
        print(f"[WARN] Файл модели не найден: {best_model_path}")
        print("Используем текущую модель для тестирования...")
        print(f"[WARN] ВНИМАНИЕ: Модель может быть не лучшей по val_accuracy!")
        test_acc, cm, report = test_model(model, test_loader, device, config.data.classes)
    else:
        print("Загрузка лучшей модели (по val_accuracy)...")
        checkpoint = torch.load(best_model_path, weights_only=False)
        saved_val_acc = checkpoint.get('val_acc', 0)
        saved_epoch = checkpoint.get('epoch', 0)
        print(f"   Загружена модель с эпохи {saved_epoch}, val_acc={saved_val_acc:.2f}%")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # ВАЖНО: устанавливаем eval режим для тестирования!
        print("Лучшая модель загружена и переведена в eval режим")
        
        print(f"\nЗапуск тестирования на {len(test_dataset)} изображениях...")
        test_acc, cm, report = test_model(model, test_loader, device, config.data.classes)
        
        # Проверяем разницу между val и test (признак переобучения)
        gap = best_val_acc - (test_acc * 100)
        print(f"\n{'='*60}")
        print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print(f"   Val Accuracy:   {best_val_acc:.2f}%")
        print(f"   Test Accuracy:  {test_acc*100:.2f}%")
        print(f"   Разница:        {gap:.2f}%")
        
        if gap > 30:
            print(f"\n[WARN] ПРОБЛЕМА: Большая разница между val и test!")
            print(f"   Это указывает на ПЕРЕОБУЧЕНИЕ (overfitting)")
            print(f"\n[TIP] Возможные причины:")
            print(f"   1. Тестовый набор слишком маленький ({len(test_dataset)} изображений)")
            print(f"   2. Модель переобучилась на train/val данных")
            print(f"   3. Недостаточная регуляризация")
            print(f"\n[TIP] Рекомендации:")
            print(f"   - Увеличьте размер тестового набора")
            print(f"   - Добавьте dropout или увеличьте weight_decay")
            print(f"   - Используйте early stopping")
            print(f"   - Уменьшите количество эпох")
        elif gap > 15:
            print(f"\n[WARN] Предупреждение: Умеренная разница между val и test")
        else:
            print(f"\n[OK] Хорошо: Небольшая разница между val и test")
    
    print(f"\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Classification Report:")
    print(f"{report}")
    
    # Сохраняем confusion matrix
    plot_confusion_matrix(cm, config.data.classes, results_dir / "confusion_matrix.png")
    
    # Сохраняем финальные графики
    save_training_plots(train_losses, val_losses, train_accs, val_accs, 
                       config.model.num_epochs, results_dir)
    
    # Сохраняем результаты
    results = {
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(best_val_acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'config': {
            'model_name': config.model.model_name,
            'num_epochs': config.model.num_epochs,
            'learning_rate': config.model.learning_rate,
            'optimizer': config.model.optimizer,
            'device': config.model.device,
            'classes': config.data.classes,
            'batch_size': config.data.batch_size,
            'experiment_name': config.experiment.experiment_name,
            'random_seed': config.experiment.random_seed
        },
        'training_time_minutes': training_time / 60,
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print(f"{'='*80}")
    print(f"Результаты сохранены в: {results_dir}")
    print(f"Лучшая модель: {best_model_path}")
    print(f"Финальная точность: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Время обучения: {training_time/60:.2f} минут")
    print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
    print(f"Размер модели: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024):.2f} MB")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*80}")
    
    if config.experiment.use_tensorboard:
        writer.close()
        print("TensorBoard логи сохранены")

    return results_dir, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели классификации фруктов')
    parser.add_argument('--model', type=str, default='resnet18', 
                       help='Название модели (resnet18, efficientnet_b0, mobilenetv3_small_100)')
    parser.add_argument('--epochs', type=int, default=20, help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--device', type=str, default='cpu', help='Устройство (cpu/cuda)')
    # Новые аргументы для freeze/unfreeze
    parser.add_argument('--freeze_epochs', type=int, default=None, help='Сколько эпох обучать только head (backbone заморожен)')
    parser.add_argument('--head_lr', type=float, default=None, help='Learning rate для head')
    parser.add_argument('--backbone_lr', type=float, default=None, help='Learning rate для backbone после разморозки')
    
    args = parser.parse_args()
    
    # Создаем конфигурацию
    config = Config()
    config.model.model_name = args.model
    config.model.num_epochs = args.epochs
    config.model.learning_rate = args.lr
    config.data.batch_size = args.batch_size
    config.model.device = args.device
    config.experiment.experiment_name = f"{args.model}_experiment"
    # Применяем CLI-переопределения freeze/unfreeze при наличии
    if args.freeze_epochs is not None:
        config.model.freeze_epochs = args.freeze_epochs
    if args.head_lr is not None:
        config.model.head_lr = args.head_lr
    if args.backbone_lr is not None:
        config.model.backbone_lr = args.backbone_lr
    
    # Запускаем обучение
    results_dir, test_acc = main(config)
