"""
Основная конфигурация проекта
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime

@dataclass
class DataConfig:
    """Конфигурация данных"""
    # Основные пути
    project_root: Path = Path(__file__).parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).parent / "experiments")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Параметры данных
    classes: List[str] = field(default_factory=lambda: ["apple", "kiwi", "mandarin"])
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    
    # Разделение данных
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Аугментация
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Архитектура модели
    model_name: str = "efficientnet_b0"  # зафиксирована лучшая модель для CPU-инференса
    pretrained: bool = True
    
    # Параметры обучения
    num_epochs: int = 20
    learning_rate: float = 0.001  # базовый lr (используется как нижняя граница)
    optimizer: str = "AdamW"      # оптимизатор по умолчанию
    
    # Стратегия transfer learning: freeze -> unfreeze
    freeze_epochs: int = 5        # сколько эпох обучать только head
    head_lr: float = 1e-3         # lr для head (на freeze-этапе и после)
    backbone_lr: float = 2e-4     # lr для backbone после разморозки
    loss_function: str = "CrossEntropyLoss"
    
    # Сохранение
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 5
    
    # Устройство
    device: str = "cpu"

@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    # Идентификация эксперимента
    experiment_name: str = "fruit_classification"
    run_name: Optional[str] = None
    
    # Воспроизводимость
    random_seed: int = 42
    
    # Логирование
    use_tensorboard: bool = True
    log_freq: int = 10
    save_plots_every_epoch: bool = True
    
    # Валидация
    val_freq: int = 1
    
    # Тестирование
    test_after_training: bool = True

@dataclass
class Config:
    """Общая конфигурация проекта"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        # Обновляем num_classes в ModelConfig на основе DataConfig
        self.model.num_classes = self.data.num_classes
        
        # Устанавливаем run_name, если не задан
        if self.experiment.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment.run_name = f"{self.model.model_name}_{timestamp}"
        
        # Обновляем пути для экспериментов и логов
        self.data.experiments_dir = self.data.experiments_dir / self.experiment.run_name
        self.data.logs_dir = self.data.logs_dir / self.experiment.run_name
        
    def to_dict(self):
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "experiment": self.experiment.__dict__
        }





