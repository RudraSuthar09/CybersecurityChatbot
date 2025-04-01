import os
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
    logger.info("Current Date and Time (UTC): 2025-03-31 04:46:12")
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

    logger.info("Dataset Status:")
    logger.info(f"NSL-KDD Train: {'Found' if nslkdd_train.exists() else 'Missing'}")
    logger.info(f"NSL-KDD Test: {'Found' if nslkdd_test.exists() else 'Missing'}")
    logger.info(f"MITRE ATT&CK: {'Found' if mitre_path.exists() else 'Missing'}")

    logger.info("Project initialization completed!")

if __name__ == "__main__":
    main()
