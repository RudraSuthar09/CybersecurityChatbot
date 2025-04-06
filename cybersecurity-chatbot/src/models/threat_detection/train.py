import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from datetime import datetime
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[3]  # Adjusting depth if needed
sys.path.append(str(project_root))

from src.data.processors.nslkdd_processor import NSLKDDProcessor
from src.models.base.network_analyzer import NetworkAnalyzer

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Model Trainer")
        self.logger.info(f"Current Date and Time (UTC): {datetime.utcnow()}")
        
        # Dynamically locate config.yaml
        config_path = project_root / "src/config/config.yaml"
        if not config_path.exists():
            self.logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Set random seed
        torch.manual_seed(self.config['training']['seed'])

        # Training parameters
        self.batch_size = self.config['model']['network_analyzer']['batch_size']
        self.learning_rate = self.config['model']['network_analyzer']['learning_rate']
        self.epochs = self.config['model']['network_analyzer']['epochs']

    def prepare_data(self):
        """Prepare data for training"""
        self.logger.info("Preparing data for training")
        processor = NSLKDDProcessor()
        X_train, X_test, y_train, y_test = processor.process()
        X_train = torch.FloatTensor(X_train.values)
        X_test = torch.FloatTensor(X_test.values)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return processor.label_encoders['attack_type']

    def train_epoch(self, model, criterion, optimizer, epoch):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        return total_loss / len(self.train_loader), 100.*correct/total

    def evaluate(self, model, criterion, label_encoder):
        model.eval()
        test_loss, correct, total = 0, 0, 0
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        accuracy = 100. * correct / total
        test_loss /= len(self.test_loader)
        cm = confusion_matrix(all_targets, all_predictions)
        return test_loss, accuracy, cm

    def train(self):
        label_encoder = self.prepare_data()
        model = NetworkAnalyzer(self.config).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        best_accuracy = 0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(model, criterion, optimizer, epoch)
            test_loss, test_acc, cm = self.evaluate(model, criterion, label_encoder)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), "best_model.pth")
        return best_accuracy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    trainer = ModelTrainer()
    best_accuracy = trainer.train()
    print(f"\nTraining completed! Best Test Accuracy: {best_accuracy:.2f}%")
