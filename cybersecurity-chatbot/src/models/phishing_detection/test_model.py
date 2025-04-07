import os
from datetime import datetime, UTC
from urllib.parse import urlparse
import torch
import numpy as np
from extract_features import URLFeatureExtractor
from models.phishing_classifier import PhishingClassifier

def load_latest_model():
    """Load the most recent trained model"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'saved_models')
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError("No saved models found!")
        
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
        model_path = os.path.join(model_dir, latest_model)
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path)
        
        model = PhishingClassifier(input_dim=checkpoint['architecture']['input_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint['training_info']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict_url(model, url, feature_extractor):
    """Make prediction for a single URL"""
    try:
        # Extract features
        features = feature_extractor.extract_features(url)
        
        # Reshape features to 2D tensor (batch_size=1, features)
        features_tensor = torch.FloatTensor(features).reshape(1, -1)
        
        with torch.no_grad():
            output = model(features_tensor)
            probability = output.item()
            prediction = 1 if probability >= 0.5 else 0
        
        return prediction, probability, features
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None, None

def save_result(url, prediction, probability, features, model_info):
    """Save prediction results"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictions_dir = os.path.join(current_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        result = {
            'timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S'),
            'url': url,
            'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
            'confidence': f"{probability*100:.2f}%",
            'features': features.tolist(),
            'model_info': {
                'training_accuracy': f"{model_info['final_train_acc']:.2f}%",
                'validation_accuracy': f"{model_info['final_val_acc']:.2f}%"
            }
        }
        
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(predictions_dir, f'prediction_{timestamp}.txt')
        
        with open(filename, 'w') as f:
            f.write("URL Phishing Detection Result\n")
            f.write("============================\n\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Confidence: {result['confidence']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n\n")
            f.write("Model Information:\n")
            f.write(f"Training Accuracy: {result['model_info']['training_accuracy']}\n")
            f.write(f"Validation Accuracy: {result['model_info']['validation_accuracy']}\n")
        
        print(f"\nResults saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return None

def main():
    print(f"Current Date and Time (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: {os.getlogin()}")
    
    # Load model
    model, model_info = load_latest_model()
    if model is None:
        return
    
    # Initialize feature extractor
    feature_extractor = URLFeatureExtractor()
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "http://suspicious-site.com/login/secure/paypal/login.html",
        "https://facebook.com",
        "http://faceb00k.com/login"
    ]
    
    print("\nTesting URLs...")
    print("=" * 50)
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        prediction, probability, features = predict_url(model, url, feature_extractor)
        
        if prediction is not None:
            result = "PHISHING" if prediction == 1 else "LEGITIMATE"
            print(f"Prediction: {result}")
            print(f"Confidence: {probability*100:.2f}%")
            
            # Save result
            save_result(url, prediction, probability, features, model_info)
    
    print("\nDo you want to test custom URLs? (yes/no)")
    if input().lower().startswith('y'):
        while True:
            print("\nEnter a URL to test (or 'quit' to exit):")
            url = input().strip()
            
            if url.lower() == 'quit':
                break
            
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            prediction, probability, features = predict_url(model, url, feature_extractor)
            
            if prediction is not None:
                result = "PHISHING" if prediction == 1 else "LEGITIMATE"
                print(f"Prediction: {result}")
                print(f"Confidence: {probability*100:.2f}%")
                
                # Save result
                save_result(url, prediction, probability, features, model_info)

if __name__ == "__main__":
    main()