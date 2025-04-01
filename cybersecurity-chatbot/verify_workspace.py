import os
from pathlib import Path
import datetime
import json

def verify_workspace():
    print(f"Workspace Verification Report")
    print(f"Current Date and Time (UTC): 2025-03-31 04:36:24")
    print(f"Current User: RudraSuthar09")
    print("=" * 50)

    # Expected project structure
    project_structure = {
        'datasets': {
            'intrusion_detection/nslkdd': ['raw', 'processed'],
            'threat_detection/mitre': ['enterprise-attack.json']
        },
        'src': {
            'data': ['processors', 'loaders'],
            'models': ['base', 'threat_detection'],
            'utils': [],
            'config': []
        },
        'tests': [],
        'logs': []
    }

    # Create project structure
    for main_dir, sub_dirs in project_structure.items():
        if isinstance(sub_dirs, list):
            for sub_dir in sub_dirs:
                Path(f"{main_dir}/{sub_dir}").mkdir(parents=True, exist_ok=True)
        else:
            for sub_path, folders in sub_dirs.items():
                for folder in folders:
                    Path(f"{main_dir}/{sub_path}/{folder}").mkdir(parents=True, exist_ok=True)

    print("\n✨ Project structure created successfully!")
    
    # Verify datasets
    nslkdd_path = Path("datasets/intrusion_detection/nslkdd/processed/KDDTrain+.csv")
    mitre_path = Path("datasets/threat_detection/mitre/enterprise-attack.json")
    
    print("\n📂 Dataset Verification:")
    print(f"NSL-KDD Dataset: {'✅ Found' if nslkdd_path.exists() else '❌ Missing'}")
    print(f"MITRE Dataset: {'✅ Found' if mitre_path.exists() else '❌ Missing'}")

if __name__ == "__main__":
    verify_workspace()