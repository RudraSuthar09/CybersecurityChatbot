import torch
import os
from datetime import datetime, UTC
import json
from models.phishing_classifier import PhishingClassifier

def load_model(model_path):
    """Load the trained model"""
    try:
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path)
        
        # Initialize model with same architecture
        model = PhishingClassifier(input_dim=checkpoint['architecture']['input_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully!")
        print(f"Model architecture: Input dim={checkpoint['architecture']['input_dim']}, "
              f"Hidden dims={checkpoint['architecture']['hidden_dims']}")
        return model, checkpoint['training_info']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict_url(model, url_features):
    """Make prediction for a single URL"""
    with torch.no_grad():
        features = torch.FloatTensor(url_features)
        output = model(features)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0
        return prediction, probability

def main():
    # Print execution info
    current_time = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
    current_user = os.getlogin()
    print(f"Current Date and Time (UTC): {current_time}")
    print(f"Current User: {current_user}")
    
    try:
        # Get the latest model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        
        if not model_files:
            print("No saved models found!")
            return
        
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
        model_path = os.path.join(model_dir, latest_model)
        
        # Load the model
        model, training_info = load_model(model_path)
        if model is None:
            return
        
        print("\nModel Information:")
        print(f"Training accuracy: {training_info['final_train_acc']:.2f}%")
        print(f"Validation accuracy: {training_info['final_val_acc']:.2f}%")
        
        # Create predictions directory if it doesn't exist
        predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        while True:
            # Get URL features from user
            print("\nEnter URL features (comma-separated values, 30 features) or 'quit' to exit:")
            user_input = input().strip()
            
            if user_input.lower() == 'quit':
                break
            
            try:
                # Parse features
                features = [float(x.strip()) for x in user_input.split(',')]
                if len(features) != 30:
                    print(f"Error: Expected 30 features, got {len(features)}")
                    continue
                
                # Make prediction
                prediction, probability = predict_url(model, features)
                
                # Print result
                result = "PHISHING" if prediction == 1 else "LEGITIMATE"
                print(f"\nPrediction: {result}")
                print(f"Confidence: {probability*100:.2f}%")
                
                # Save prediction
                prediction_data = {
                    'timestamp': current_time,
                    'user': current_user,
                    'features': features,
                    'prediction': result,
                    'probability': probability,
                    'model_file': latest_model
                }
                
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                prediction_file = os.path.join(predictions_dir, f'prediction_{timestamp}.json')
                
                with open(prediction_file, 'w') as f:
                    json.dump(prediction_data, f, indent=4)
                
                print(f"Prediction saved to: {prediction_file}")
                
            except ValueError:
                print("Error: Invalid input format. Please enter comma-separated numbers.")
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()