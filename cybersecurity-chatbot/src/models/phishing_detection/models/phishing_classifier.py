import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import getpass

# Print execution info
current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
current_user = getpass.getuser()
print(f"Current Date and Time (UTC): {current_time}")
print(f"Current User: {current_user}")

class PhishingClassifier(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64, 32, 16]):
        super(PhishingClassifier, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        # Create sequential model for hidden layers
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Initialized PhishingClassifier with:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Hidden dimensions: {hidden_dims}")
    
    def _initialize_weights(self):
        """Initialize model weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Apply hidden layers
        x = self.hidden_layers(x)
        
        # Apply output layer with sigmoid activation
        x = torch.sigmoid(self.output(x))
        return x
    
    def predict(self, x):
        """Predict probabilities for input data"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            probabilities = self.forward(x)
        return probabilities
    
    def get_feature_importance(self, x):
        """Calculate feature importance using gradient-based approach"""
        x.requires_grad_(True)
        self.eval()
        
        # Forward pass
        output = self.forward(x)
        
        # Calculate gradients
        output.backward(torch.ones_like(output))
        
        # Get feature importance from gradients
        importance = torch.abs(x.grad).mean(dim=0)
        return importance

def test_model():
    """Test the model with sample data"""
    print("\nTesting PhishingClassifier...")
    
    # Create sample data
    input_dim = 9
    batch_size = 4
    sample_input = torch.randn(batch_size, input_dim)
    
    # Initialize model
    model = PhishingClassifier(input_dim=input_dim)
    
    # Test forward pass
    output = model(sample_input)
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values:\n{output.detach().numpy()}")
    
    # Test feature importance
    importance = model.get_feature_importance(sample_input)
    print(f"\nFeature importance values:\n{importance.detach().numpy()}")

if __name__ == "__main__":
    test_model()