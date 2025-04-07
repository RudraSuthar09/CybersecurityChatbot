import numpy as np
from urllib.parse import urlparse
import re

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        pass
    
    def extract_features(self, url):
        """Extract features from a single URL"""
        features = []
        
        # Basic URL properties
        features.append(len(url))  # URL length
        features.append(url.count('/'))  # Number of forward slashes
        features.append(url.count('.'))  # Number of dots
        features.append(url.count('?'))  # Number of question marks
        features.append(url.count('='))  # Number of equals signs
        features.append(url.count('-'))  # Number of hyphens
        features.append(url.count('_'))  # Number of underscores
        
        # Parse URL
        parsed = urlparse(url)
        
        # Domain-based features
        domain = parsed.netloc
        features.append(len(domain))  # Domain length
        features.append(domain.count('.'))  # Number of dots in domain
        features.append(domain.count('-'))  # Number of hyphens in domain
        
        # Path-based features
        path = parsed.path
        features.append(len(path))  # Path length
        features.append(path.count('/'))  # Number of directories
        
        # Fill remaining features with zeros (placeholder)
        # In a real implementation, you'd extract more meaningful features
        while len(features) < 30:
            features.append(0)
        
        return np.array(features)

def main():
    extractor = URLFeatureExtractor()
    
    while True:
        print("\nEnter a URL to extract features (or 'quit' to exit):")
        url = input().strip()
        
        if url.lower() == 'quit':
            break
        
        try:
            features = extractor.extract_features(url)
            print("\nExtracted features:")
            print(','.join(map(str, features)))
        except Exception as e:
            print(f"Error extracting features: {str(e)}")

if __name__ == "__main__":
    main()