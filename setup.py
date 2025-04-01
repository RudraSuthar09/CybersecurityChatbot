import os
from pathlib import Path
import datetime

def create_structure():
    # Your current location
    base_path = r"D:\Chatbot datasets\cybersecurity-chatbot"
    
    print(f"Setting up project structure at: {base_path}")
    print(f"Current Date and Time (UTC): 2025-03-31 04:43:24")
    print(f"Current User: RudraSuthar09")
    
    # Create directories
    directories = [
        r"src\config",
        r"src\data\processors",
        r"src\data\loaders",
        r"src\models\base",
        r"src\models\threat_detection",
        r"src\utils",
        "tests",
        "logs"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        Path(full_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

    # Create config.yaml
    config_path = os.path.join(base_path, "src", "config", "config.yaml")
    config_content = """# Project Configuration
project:
  name: "Cybersecurity Chatbot"
  version: "1.0.0"
  author: "RudraSuthar09"
  created_at: "2025-03-31 04:43:24"

# Data Configuration
data:
  nslkdd:
    train_path: "datasets/intrusion_detection/nslkdd/processed/KDDTrain+.csv"
    test_path: "datasets/intrusion_detection/nslkdd/processed/KDDTest+.csv"
    features: 41
    classes: 5

  mitre:
    path: "datasets/threat_detection/mitre/enterprise-attack.json"

# Model Configuration
model:
  network_analyzer:
    input_size: 41
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    batch_size: 64
    learning_rate: 0.001
    epochs: 10

# Training Configuration
training:
  device: "cuda" if torch.cuda.is_available() else "cpu"
  seed: 42
  validation_split: 0.2

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/cybersec_chatbot.log"
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"✅ Created configuration file: src/config/config.yaml")

    # Create initialize_project.py
    init_path = os.path.join(base_path, "initialize_project.py")
    init_content = """import os
import sys
from pathlib import Path
import logging
import yaml
import torch
import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Project Initialization Started")
    logger.info("Current Date and Time (UTC): 2025-03-31 04:43:24")
    logger.info("Current User: RudraSuthar09")

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

    logger.info("\\nDataset Status:")
    logger.info(f"NSL-KDD Train: {'✅ Found' if nslkdd_train.exists() else '❌ Missing'}")
    logger.info(f"NSL-KDD Test: {'✅ Found' if nslkdd_test.exists() else '❌ Missing'}")
    logger.info(f"MITRE ATT&CK: {'✅ Found' if mitre_path.exists() else '❌ Missing'}")

    logger.info("\\nProject initialization completed!")

if __name__ == "__main__":
    main()
"""
    
    with open(init_path, "w") as f:
        f.write(init_content)
    print(f"✅ Created initialization script: initialize_project.py")

if __name__ == "__main__":
    create_structure()
    print("\n✅ Setup completed! Your project structure is ready.")
    print("\nTo get started:")
    print("1. Run: python initialize_project.py")
    print("2. Check the logs folder for setup.log")