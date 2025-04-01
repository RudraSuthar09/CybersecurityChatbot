import json
from pathlib import Path
from datetime import datetime

def save_training_metrics(metrics, model_name="network_analyzer"):
    """Save training metrics to a JSON file"""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_file = metrics_dir / f"{model_name}_metrics_{timestamp}.json"
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)