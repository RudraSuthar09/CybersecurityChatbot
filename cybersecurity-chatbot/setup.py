import os
import sys
from pathlib import Path
import logging
import yaml
import torch
import datetime

def setup_logging():
    """Configure logging for the project"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def verify_dependencies():
    """Verify all required packages are installed"""
    required_packages = {
        'torch': torch.__version__,
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            if package == 'torch':
                continue  # Already imported
            else:
                __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    # Setup logging
    logger = setup_logging()
    logger.info(f"Project Initialization Started")
    logger.info(f"Current Date and Time (UTC): 2025-03-31 04:36:24")
    logger.info(f"Current User: RudraSuthar09")

    # Verify dependencies
    missing_packages = verify_dependencies()
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        sys.exit(1)

    # Load configuration
    try:
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

    # Verify dataset paths
    nslkdd_train = Path(config['data']['nslkdd']['train_path'])
    nslkdd_test = Path(config['data']['nslkdd']['test_path'])
    mitre_path = Path(config['data']['mitre']['path'])

    dataset_status = {
        'NSL-KDD Train': nslkdd_train.exists(),
        'NSL-KDD Test': nslkdd_test.exists(),
        'MITRE ATT&CK': mitre_path.exists()
    }

    logger.info("\nDataset Status:")
    for dataset, exists in dataset_status.items():
        logger.info(f"{dataset}: {'✅ Found' if exists else '❌ Missing'}")

    # Check CUDA availability
    logger.info(f"\nCUDA Status:")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"Using Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    logger.info("\nProject initialization completed!")

if __name__ == "__main__":
    main()