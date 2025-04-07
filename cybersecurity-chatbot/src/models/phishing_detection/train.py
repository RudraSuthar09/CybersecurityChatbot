import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, UTC  # Changed this line
import getpass
import requests  

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Try relative import first
    from .models.phishing_classifier import PhishingClassifier
except ImportError:
    # If relative import fails, try absolute import
    from src.models.phishing_detection.models.phishing_classifier import PhishingClassifier


class PhishingDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_uci_dataset():
    """Download and prepare UCI Phishing Websites Dataset"""
    print("Downloading UCI Phishing Websites Dataset...")
    
    try:
        # URL for the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
        
        # Read the ARFF file manually since it's not a standard CSV
        import requests
        from io import StringIO
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Convert ARFF to DataFrame
        data_lines = []
        for line in StringIO(response.text):
            # Skip comments and @attribute lines
            if not line.startswith('@'):
                # Remove any trailing comments and strip whitespace
                line = line.split('%')[0].strip()
                if line:  # Skip empty lines
                    data_lines.append(line)
        
        # Convert to DataFrame
        data = pd.DataFrame([row.split(',') for row in data_lines], dtype=float)
        
        # Features are all columns except the last one
        X = data.iloc[:, :-1]
        
        # Target is the last column (-1 for legitimate, 1 for phishing)
        y = data.iloc[:, -1].map({-1.0: 0, 1.0: 1})  # Convert to 0 and 1
        
        print("\nDataset Summary:")
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Phishing URLs: {sum(y == 1)}")
        print(f"Legitimate URLs: {sum(y == 0)}")
        
        return X, y
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise

def save_model(model, optimizer, history, X, current_time, current_user):
    """Save the trained model and related data"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create directories if they don't exist
        save_dir = os.path.join(current_dir, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_filename = f'phishing_model_{timestamp}.pt'
        save_path = os.path.join(save_dir, model_filename)
        
        # Prepare data to save
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'architecture': {
                'input_dim': model.input_dim,
                'hidden_dims': model.hidden_dims
            },
            'training_info': {
                'timestamp': timestamp,
                'epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.001,
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'dataset_size': len(X),
                'features': X.shape[1]
            },
            'user_info': {
                'timestamp': current_time,
                'user': current_user
            }
        }
        
        # Save the model
        torch.save(save_data, save_path)
        print(f"\nModel saved successfully to: {save_path}")
        
        # Save model info in text format
        info_filename = os.path.join(save_dir, f'model_info_{timestamp}.txt')
        with open(info_filename, 'w') as f:
            f.write(f"Model Information:\n")
            f.write(f"=====================================\n")
            f.write(f"Saved on: {timestamp}\n")
            f.write(f"By user: {current_user}\n")
            f.write(f"\nArchitecture:\n")
            f.write(f"Input dimensions: {model.input_dim}\n")
            f.write(f"Hidden dimensions: {model.hidden_dims}\n")
            f.write(f"\nPerformance:\n")
            f.write(f"Final training accuracy: {history['train_acc'][-1]:.2f}%\n")
            f.write(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%\n")
            f.write(f"\nDataset:\n")
            f.write(f"Total samples: {len(X)}\n")
            f.write(f"Number of features: {X.shape[1]}\n")
        
        print(f"Model info saved to: {info_filename}")
        return save_path
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions = (outputs >= 0.5).float()
            all_preds.extend(predictions.numpy())
            all_labels.extend(batch_y.numpy())
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()




def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Train the model and return training history"""
    print("\nStarting Training...")
    print("-" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted.reshape(-1) == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.reshape(-1, 1))
                val_loss += loss.item()
                predicted = (outputs.data > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted.reshape(-1) == batch_y).sum().item()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Store the metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print("-" * 60)
    
    return history

def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get current time and user
    current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    current_user = os.getlogin()
    print(f"Current Date and Time (UTC): {current_time}")
    print(f"Current User: {current_user}")
    
    try:
        print("\nLoading UCI Phishing Websites Dataset...")
        X, y = load_uci_dataset()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Create datasets
        train_dataset = PhishingDataset(X_train, y_train)
        val_dataset = PhishingDataset(X_val, y_val)
        test_dataset = PhishingDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model
        model = PhishingClassifier(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
        
        # Save the model
        save_path = save_model(model, optimizer, history, X, current_time, current_user)
        if save_path is None:
            print("Failed to save model!")
            return
        
        # Create plots directory and save training history plot
        plots_dir = os.path.join(current_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        # Save the plot
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f'training_history_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training history plot saved to: {plot_path}")
        
        # Evaluate the model
        print("\nEvaluating model on test set...")
        evaluate_model(model, test_loader)
        
        # Verify saved model
        print("\nVerifying saved model...")
        try:
            loaded_data = torch.load(save_path)
            print("Model verification successful!")
            print("Saved model contains:")
            for key in loaded_data.keys():
                print(f"- {key}")
        except Exception as e:
            print(f"Error verifying saved model: {str(e)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check internet connection and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
