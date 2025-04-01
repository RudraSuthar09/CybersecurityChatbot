import sys
import unittest
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.processors.nslkdd_processor import NSLKDDProcessor

class TestNSLKDDProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test case - called before tests in class are run"""
        print("Setting up NSL-KDD Processor Tests")
        print("Current Date and Time (UTC): 2025-03-31 05:12:59")
        print("Current User: RudraSuthar09")
        print("=" * 50)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        cls.processor = NSLKDDProcessor()

    def test_data_loading(self):
        """Test data loading functionality"""
        train_data, test_data = self.processor.load_data()
        
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(test_data)
        self.assertEqual(train_data.shape[1], 43)  # 42 features + 1 label
        self.assertEqual(test_data.shape[1], 43)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

    def test_data_processing(self):
        """Test complete data processing pipeline"""
        X_train, X_test, y_train, y_test = self.processor.process()
        
        # Verify shapes
        self.assertEqual(X_train.shape[1], 42)  # 42 features
        self.assertEqual(X_test.shape[1], 42)
        self.assertEqual(y_train.shape[0], X_train.shape[0])
        self.assertEqual(y_test.shape[0], X_test.shape[0])
        
        # Verify no missing values
        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())
        
        print("Data Processing Results:")
        print(f"Training Features Shape: {X_train.shape}")
        print(f"Test Features Shape: {X_test.shape}")
        print(f"Training Labels Shape: {y_train.shape}")
        print(f"Test Labels Shape: {y_test.shape}")
        
        # Verify attack types
        unique_attacks_train = len(set(y_train))
        unique_attacks_test = len(set(y_test))
        print(f"Unique attack types in training: {unique_attacks_train}")
        print(f"Unique attack types in test: {unique_attacks_test}")

    def test_attack_distribution(self):
        """Test attack type distribution analysis"""
        X_train, X_test, y_train, y_test = self.processor.process()
        
        # Get distributions
        train_dist = self.processor.get_attack_distribution(y_train)
        test_dist = self.processor.get_attack_distribution(y_test)
        
        # Verify normal traffic is present in both sets
        self.assertIn('normal', train_dist.index)
        self.assertIn('normal', test_dist.index)
        
        # Verify unknown attacks are handled
        self.assertIn('unknown', test_dist.index)
        
        print("Attack Distribution Test Results:")
        print("\nTraining Data Distribution:")
        print(train_dist)
        print("\nTest Data Distribution:")
        print(test_dist)

if __name__ == '__main__':
    unittest.main(verbosity=2)