import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import yaml
from datetime import datetime

class NSLKDDProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing NSL-KDD Processor")
        self.logger.info(f"Current Date and Time (UTC): 2025-03-31 05:04:02")
        self.logger.info(f"Current User: RudraSuthar09")
        
        # Load configuration
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'))
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize preprocessing tools
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Define feature groups
        self.categorical_features = ['protocol_type', 'service', 'flag']
        self.numeric_features = None
        
    def load_data(self):
        try:
            train_path = self.config['data']['nslkdd']['train_path']
            test_path = self.config['data']['nslkdd']['test_path']
            
            self.logger.info(f"Loading training data from: {train_path}")
            train_data = pd.read_csv(train_path)
            
            self.logger.info(f"Loading test data from: {test_path}")
            test_data = pd.read_csv(test_path)
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def handle_unseen_labels(self, train_data, test_data, column):
        """Handle unseen labels in test data by mapping them to 'unknown'"""
        # Get unique values from both datasets
        train_values = set(train_data[column].unique())
        test_values = set(test_data[column].unique())
        
        # Find unseen labels
        unseen_labels = test_values - train_values
        if unseen_labels:
            self.logger.warning(f"Found unseen labels in {column}: {unseen_labels}")
            
            # Map unseen labels to 'unknown'
            test_data[column] = test_data[column].apply(
                lambda x: 'unknown' if x in unseen_labels else x
            )
            
            # Add 'unknown' to training data if not present
            if 'unknown' not in train_values:
                # Add a small number of 'unknown' samples to training
                train_data.loc[len(train_data)] = train_data.iloc[0]
                train_data.iloc[-1, train_data.columns.get_loc(column)] = 'unknown'
        
        return train_data, test_data
            
    def preprocess_data(self, train_data, test_data):
        try:
            self.logger.info("Starting data preprocessing")
            
            # Handle unseen labels in categorical features
            for feature in self.categorical_features:
                self.logger.info(f"Processing categorical feature: {feature}")
                train_data, test_data = self.handle_unseen_labels(
                    train_data, test_data, feature
                )
                
                # Encode categorical features
                le = LabelEncoder()
                train_data[feature] = le.fit_transform(train_data[feature])
                test_data[feature] = le.transform(test_data[feature])
                self.label_encoders[feature] = le
            
            # Handle unseen attack types
            self.logger.info("Processing attack types")
            train_data, test_data = self.handle_unseen_labels(
                train_data, test_data, 'attack_type'
            )
            
            # Identify numeric features
            self.numeric_features = train_data.select_dtypes(
                include=['float64', 'int64']
            ).columns.tolist()
            
            # Scale numeric features
            self.logger.info("Scaling numeric features")
            train_data[self.numeric_features] = self.scaler.fit_transform(
                train_data[self.numeric_features]
            )
            test_data[self.numeric_features] = self.scaler.transform(
                test_data[self.numeric_features]
            )
            
            # Encode labels (attack_type)
            self.logger.info("Encoding labels")
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_data['attack_type'])
            test_labels = label_encoder.transform(test_data['attack_type'])
            
            # Store label encoder for later use
            self.label_encoders['attack_type'] = label_encoder
            
            # Create feature matrices and label vectors
            X_train = train_data.drop('attack_type', axis=1)
            X_test = test_data.drop('attack_type', axis=1)
            y_train = train_labels
            y_test = test_labels
            
            self.logger.info("Data preprocessing completed successfully")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise
            
    def get_attack_distribution(self, labels, dataset_name=""):
        """Get the distribution of attack types"""
        attack_types = self.label_encoders['attack_type'].inverse_transform(labels)
        distribution = pd.Series(attack_types).value_counts()
        
        self.logger.info(f"\n{dataset_name} Attack Distribution:")
        for attack_type, count in distribution.items():
            self.logger.info(f"{attack_type}: {count}")
            
        return distribution
    
    def process(self):
        self.logger.info("Starting NSL-KDD processing pipeline")
        
        # Load data
        train_data, test_data = self.load_data()
        self.logger.info(f"Loaded training data shape: {train_data.shape}")
        self.logger.info(f"Loaded test data shape: {test_data.shape}")
        
        # Log initial attack distributions
        self.logger.info("\nInitial Attack Distributions:")
        train_attacks = train_data['attack_type'].value_counts()
        test_attacks = test_data['attack_type'].value_counts()
        
        self.logger.info("\nTraining Data:")
        for attack, count in train_attacks.items():
            self.logger.info(f"{attack}: {count}")
            
        self.logger.info("\nTest Data:")
        for attack, count in test_attacks.items():
            self.logger.info(f"{attack}: {count}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(train_data, test_data)
        
        # Get and log final attack distributions
        self.get_attack_distribution(y_train, "Training Data")
        self.get_attack_distribution(y_test, "Test Data")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run processor
    processor = NSLKDDProcessor()
    X_train, X_test, y_train, y_test = processor.process()
    
    print("\nProcessing Summary:")
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Test Features Shape: {X_test.shape}")
    print(f"Training Labels Shape: {y_train.shape}")
    print(f"Test Labels Shape: {y_test.shape}")