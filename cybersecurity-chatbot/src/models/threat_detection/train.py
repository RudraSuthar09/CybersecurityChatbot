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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.processors.nslkdd_processor import NSLKDDProcessor
from src.models.base.network_analyzer import NetworkAnalyzer

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Model Trainer")
        self.logger.info(f"Current Date and Time (UTC): 2025-03-31 07:20:11")
        self.logger.info(f"Current User: RudraSuthar09")
        
        # Load configuration
        with open('src/config/config.yaml', 'r') as f:
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
        
        # Add learning rate scheduler parameters
        self.scheduler_patience = 2
        self.scheduler_factor = 0.5
        
    def prepare_data(self):
        """Prepare data for training"""
        self.logger.info("Preparing data for training")
        
        # Process data
        processor = NSLKDDProcessor()
        X_train, X_test, y_train, y_test = processor.process()
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train.values)
        X_test = torch.FloatTensor(X_test.values)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size
        )
        
        return processor.label_encoders['attack_type']
        
    def train_epoch(self, model, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Learning rate warmup
            if epoch < 5:  # 5 epochs warmup
                warmup_lr = self.learning_rate * min(1., (epoch * len(self.train_loader) + batch_idx + 1) / 
                                                   (5 * len(self.train_loader)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Accuracy: {100.*correct/total:.2f}%, '
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
                )
                
        return total_loss / len(self.train_loader), 100.*correct/total
        
    def evaluate(self, model, criterion, label_encoder):
        """Evaluate the model"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
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
        
        # Calculate metrics
        accuracy = 100.*correct/total
        test_loss /= len(self.test_loader)
        
        # Get unique classes actually present in the data
        unique_classes = sorted(set(all_targets))
        
        # Filter label names to match actual classes
        target_names = [label_encoder.classes_[i] for i in unique_classes]
        
        # Generate classification report
        try:
            report = classification_report(
                all_targets,
                all_predictions,
                target_names=target_names,
                zero_division=0
            )
        except ValueError as e:
            self.logger.warning(f"Error in classification report: {str(e)}")
            self.logger.warning("Generating report without target names...")
            report = classification_report(
                all_targets,
                all_predictions,
                zero_division=0
            )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return test_loss, accuracy, report, cm
        
    def calculate_class_weights(self, y_train):
        """Calculate class weights to handle class imbalance"""
        unique, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        # Calculate weights as inverse of frequency
        weights = torch.FloatTensor(
            [total_samples / (len(unique) * count) for count in counts]
        )
        
        self.logger.info("\nClass Weights:")
        for cls, weight in zip(unique, weights):
            self.logger.info(f"Class {cls}: {weight:.4f}")
        
        return weights.to(self.device)
        
    def train(self):
        """Complete training pipeline"""
        self.logger.info("Starting training pipeline")
        
        # Prepare data
        label_encoder = self.prepare_data()
        
        # Calculate class weights if enabled
        if self.config['training'].get('class_weights', False):
            _, _, y_train, _ = NSLKDDProcessor().process()
            class_weights = self.calculate_class_weights(y_train)
        else:
            class_weights = None
        
        # Initialize model
        model = NetworkAnalyzer(self.config).to(self.device)
        
        # Use weighted loss if class weights are enabled
        criterion = (
            nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            verbose=True
        )
        
        # Training loop
        best_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nEpoch: {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, criterion, optimizer, epoch)
            
            # Evaluate
            test_loss, test_acc, report, cm = self.evaluate(
                model, criterion, label_encoder
            )
            
            # Learning rate scheduling
            scheduler.step(test_acc)
            
            # Early stopping
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                patience_counter = 0
                Path('models').mkdir(exist_ok=True)
                torch.save(
                    model.state_dict(), 
                    'models/best_model.pth'
                )
                self.logger.info(
                    f"New best model saved with accuracy: {test_acc:.2f}%"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info(
                        f"Early stopping triggered after {epoch+1} epochs"
                    )
                    break
            
            # Log results
            self.logger.info(f"\nTraining Loss: {train_loss:.4f}")
            self.logger.info(f"Training Accuracy: {train_acc:.2f}%")
            self.logger.info(f"Test Loss: {test_loss:.4f}")
            self.logger.info(f"Test Accuracy: {test_acc:.2f}%")
            self.logger.info("\nClassification Report:")
            self.logger.info(f"\n{report}")
            
        return best_accuracy

    def plot_confusion_matrix(self, cm, label_encoder, epoch):
        """Plot confusion matrix"""
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Save plot
        Path('plots').mkdir(exist_ok=True)
        plt.savefig(f'plots/confusion_matrix_epoch_{epoch}.png')
        plt.close()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train model
    trainer = ModelTrainer()
    best_accuracy = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")