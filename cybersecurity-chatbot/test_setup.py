from pathlib import Path
import sys
import logging
from datetime import datetime

def verify_setup():
    # Required directories
    required_dirs = [
        "src/utils",
        "src/config",
        "src/models/base",
        "src/models/threat_detection",
        "metrics",
        "plots",
        "models"
    ]
    
    # Required files
    required_files = [
        "src/config/config.yaml",
        "src/utils/metrics.py",
        "src/models/base/network_analyzer.py",
        "src/models/threat_detection/train.py"
    ]
    
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: RudraSuthar09")
    print("\nVerifying project setup...")
    
    # Check directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Verify files exist
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
    else:
        print("\nAll required files are present!")
    
    print("\nSetup verification complete!")

if __name__ == "__main__":
    verify_setup()