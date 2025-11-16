"""
CSV Manager for local file-based storage.
Replaces Supabase database with CSV files.
"""

import csv
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import shutil


class CSVManager:
    """Manage data persistence using CSV files."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        self.models_dir = self.data_dir / "models"
        self.datasets_dir = self.data_dir / "datasets"
        self.outputs_dir = self.data_dir / "outputs"
        self.backups_dir = self.data_dir / "backups"
        self.cache_dir = self.data_dir / "cache"
        self.logs_dir = self.data_dir / "logs"

        self._ensure_directories()
        self._initialize_csv_files()

    def _ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        for directory in [
            self.metadata_dir,
            self.models_dir,
            self.datasets_dir,
            self.outputs_dir,
            self.backups_dir,
            self.cache_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        models_csv = self.metadata_dir / "models.csv"
        if not models_csv.exists():
            pd.DataFrame(columns=[
                'id', 'name', 'architecture_type', 'backbone', 'parameters_json',
                'performance_metrics', 'file_path', 'file_size_mb', 'is_pretrained',
                'created_at'
            ]).to_csv(models_csv, index=False)

        datasets_csv = self.metadata_dir / "datasets.csv"
        if not datasets_csv.exists():
            pd.DataFrame(columns=[
                'id', 'name', 'description', 'num_images', 'train_split',
                'status', 'created_at'
            ]).to_csv(datasets_csv, index=False)

        dataset_images_csv = self.metadata_dir / "dataset_images.csv"
        if not dataset_images_csv.exists():
            pd.DataFrame(columns=[
                'id', 'dataset_id', 'input_path', 'target_path', 'split_type',
                'width', 'height', 'original_filename'
            ]).to_csv(dataset_images_csv, index=False)

        training_runs_csv = self.metadata_dir / "training_runs.csv"
        if not training_runs_csv.exists():
            pd.DataFrame(columns=[
                'id', 'model_id', 'dataset_id', 'epochs', 'learning_rate',
                'batch_size', 'optimizer', 'status', 'best_epoch',
                'loss_history', 'metrics_history', 'started_at', 'completed_at'
            ]).to_csv(training_runs_csv, index=False)

    def _backup_csv(self, csv_path: Path):
        """Create a timestamped backup of a CSV file."""
        if csv_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backups_dir / f"{csv_path.stem}_{timestamp}.csv"
            shutil.copy2(csv_path, backup_path)

    def _generate_id(self, csv_path: Path) -> str:
        """Generate a unique ID for a new record."""
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return "1"
            return str(int(df['id'].max()) + 1)
        except:
            return "1"

    # Model Management
    def add_model(self, name: str, architecture_type: str, backbone: str,
                  parameters_json: Dict, performance_metrics: Dict,
                  file_path: str, file_size_mb: float,
                  is_pretrained: bool = False) -> Dict:
        """Add a new model to models.csv."""
        csv_path = self.metadata_dir / "models.csv"
        self._backup_csv(csv_path)

        model_id = self._generate_id(csv_path)
        new_model = {
            'id': model_id,
            'name': name,
            'architecture_type': architecture_type,
            'backbone': backbone,
            'parameters_json': json.dumps(parameters_json),
            'performance_metrics': json.dumps(performance_metrics),
            'file_path': file_path,
            'file_size_mb': file_size_mb,
            'is_pretrained': is_pretrained,
            'created_at': datetime.now().isoformat()
        }

        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_model])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return new_model

    def list_models(self) -> List[Dict]:
        """List all models from models.csv."""
        csv_path = self.metadata_dir / "models.csv"
        df = pd.read_csv(csv_path)

        models = []
        for _, row in df.iterrows():
            model = row.to_dict()
            model['parameters_json'] = json.loads(model['parameters_json']) if model['parameters_json'] else {}
            model['performance_metrics'] = json.loads(model['performance_metrics']) if model['performance_metrics'] else {}
            models.append(model)

        return models

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get a specific model by ID."""
        models = self.list_models()
        for model in models:
            if str(model['id']) == str(model_id):
                return model
        return None

    def update_model(self, model_id: str, updates: Dict):
        """Update a model's information."""
        csv_path = self.metadata_dir / "models.csv"
        self._backup_csv(csv_path)

        df = pd.read_csv(csv_path)
        mask = df['id'] == int(model_id)

        for key, value in updates.items():
            if key in df.columns:
                if isinstance(value, (dict, list)):
                    df.loc[mask, key] = json.dumps(value)
                else:
                    df.loc[mask, key] = value

        df.to_csv(csv_path, index=False)

    def delete_model(self, model_id: str):
        """Delete a model and its associated file."""
        model = self.get_model(model_id)
        if model and model['file_path']:
            file_path = Path(model['file_path'])
            if file_path.exists():
                file_path.unlink()

        csv_path = self.metadata_dir / "models.csv"
        self._backup_csv(csv_path)

        df = pd.read_csv(csv_path)
        df = df[df['id'] != int(model_id)]
        df.to_csv(csv_path, index=False)

    # Dataset Management
    def create_dataset(self, name: str, description: str, train_split: float = 0.8) -> Dict:
        """Create a new dataset."""
        csv_path = self.metadata_dir / "datasets.csv"
        self._backup_csv(csv_path)

        dataset_id = self._generate_id(csv_path)
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "input").mkdir(exist_ok=True)
        (dataset_dir / "target").mkdir(exist_ok=True)

        new_dataset = {
            'id': dataset_id,
            'name': name,
            'description': description,
            'num_images': 0,
            'train_split': train_split,
            'status': 'creating',
            'created_at': datetime.now().isoformat()
        }

        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_dataset])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return new_dataset

    def list_datasets(self, status: Optional[str] = None) -> List[Dict]:
        """List all datasets."""
        csv_path = self.metadata_dir / "datasets.csv"
        df = pd.read_csv(csv_path)

        if status:
            df = df[df['status'] == status]

        return df.to_dict('records')

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get a specific dataset by ID."""
        datasets = self.list_datasets()
        for dataset in datasets:
            if str(dataset['id']) == str(dataset_id):
                return dataset
        return None

    def update_dataset(self, dataset_id: str, updates: Dict):
        """Update a dataset's information."""
        csv_path = self.metadata_dir / "datasets.csv"
        self._backup_csv(csv_path)

        df = pd.read_csv(csv_path)
        mask = df['id'] == int(dataset_id)

        for key, value in updates.items():
            if key in df.columns:
                df.loc[mask, key] = value

        df.to_csv(csv_path, index=False)

    def add_dataset_image(self, dataset_id: str, input_path: str, target_path: str,
                         original_filename: str, width: int, height: int,
                         split_type: str) -> Dict:
        """Add an image pair to a dataset."""
        csv_path = self.metadata_dir / "dataset_images.csv"
        self._backup_csv(csv_path)

        image_id = self._generate_id(csv_path)
        new_image = {
            'id': image_id,
            'dataset_id': dataset_id,
            'input_path': input_path,
            'target_path': target_path,
            'split_type': split_type,
            'width': width,
            'height': height,
            'original_filename': original_filename
        }

        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_image])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return new_image

    def get_dataset_images(self, dataset_id: str, split_type: Optional[str] = None) -> List[Dict]:
        """Get all images for a dataset."""
        csv_path = self.metadata_dir / "dataset_images.csv"
        df = pd.read_csv(csv_path)

        df = df[df['dataset_id'] == int(dataset_id)]
        if split_type:
            df = df[df['split_type'] == split_type]

        return df.to_dict('records')

    # Training Run Management
    def create_training_run(self, dataset_id: str, epochs: int, learning_rate: float,
                           batch_size: int, optimizer: str) -> Dict:
        """Create a new training run record."""
        csv_path = self.metadata_dir / "training_runs.csv"
        self._backup_csv(csv_path)

        run_id = self._generate_id(csv_path)
        new_run = {
            'id': run_id,
            'model_id': '',
            'dataset_id': dataset_id,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'status': 'running',
            'best_epoch': 0,
            'loss_history': '[]',
            'metrics_history': '[]',
            'started_at': datetime.now().isoformat(),
            'completed_at': ''
        }

        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_run])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return new_run

    def update_training_run(self, run_id: str, updates: Dict):
        """Update a training run's information."""
        csv_path = self.metadata_dir / "training_runs.csv"
        self._backup_csv(csv_path)

        df = pd.read_csv(csv_path)
        mask = df['id'] == int(run_id)

        for key, value in updates.items():
            if key in df.columns:
                if isinstance(value, (dict, list)):
                    df.loc[mask, key] = json.dumps(value)
                else:
                    df.loc[mask, key] = value

        df.to_csv(csv_path, index=False)

    # Processing History
    def log_processing(self, model_id: Optional[str], parameters: Dict,
                      num_paths: int, num_points: int, processing_time_ms: int):
        """Log a processing operation to history file."""
        log_file = self.logs_dir / "processing_history.txt"

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'parameters': parameters,
            'num_paths': num_paths,
            'num_points': num_points,
            'processing_time_ms': processing_time_ms
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
