"""
Конфигурационные дата-классы для проекта fine-tuning
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import os

@dataclass
class DataConfig:
    """Конфигурация данных"""
    # Основные пути
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "experiments")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
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
    model_name: str = "resnet18"  # resnet18, resnet34, resnet50, efficientnet_b0, vit_base_patch16_224, mobilenetv3_small_100
    pretrained: bool = True
    
    # Параметры обучения
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Сохранение
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 5
    
    # Устройство
    device: str = field(default_factory=lambda: "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

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
        """Создание необходимых директорий"""
        for dir_path in [
            self.data.models_dir,
            self.data.experiments_dir,
            self.data.logs_dir,
            self.data.processed_data_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Преобразование конфигурации в словарь"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save_config(self, path: Path):
        """Сохранение конфигурации в файл"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def load_config(cls, path: Path):
        """Загрузка конфигурации из файла"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        experiment_config = ExperimentConfig(**config_dict['experiment'])
        
        return cls(data=data_config, model=model_config, experiment=experiment_config)










