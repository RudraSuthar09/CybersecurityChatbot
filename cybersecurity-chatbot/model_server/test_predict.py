import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
from src.data.processors.nslkdd_processor import NSLKDDProcessor

# Step 1: Prepare a real test sample
processor = NSLKDDProcessor()
X_train, X_test, y_train, y_test = processor.process()

# Step 2: Pick one test sample
sample = X_test.iloc[0].tolist()

# Step 3: Send to model server
response = requests.post("http://localhost:8000/predict", json={"features": sample})

print("✅ Prediction Response:", response.json())
