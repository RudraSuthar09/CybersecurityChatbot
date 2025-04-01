def create_config():
    config_content = """# Project Configuration
project:
  name: "Cybersecurity Chatbot"
  version: "1.0.0"
  author: "RudraSuthar09"
  created_at: "2025-03-31 04:57:53"

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

  threat_detector:
    embedding_dim: 256
    num_heads: 4
    num_layers: 2
    dropout: 0.1

# Training Configuration
training:
  seed: 42
  validation_split: 0.2
  early_stopping_patience: 5
  device: "cpu"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/cybersec_chatbot.log"
"""
    
    # Write the corrected configuration
    with open("src/config/config.yaml", "w") as f:
        f.write(config_content)
    
    print("Configuration file has been fixed!")

if __name__ == "__main__":
    create_config()