import re
from urllib.parse import urlparse
import numpy as np
from sklearn.preprocessing import StandardScaler

class URLFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'url_length', 'num_special_chars', 'has_ip_address',
            'num_digits', 'has_suspicious_words', 'domain_length',
            'path_length', 'num_subdomains', 'has_https'
        ]
    
    def extract_features(self, url):
        """Extract features from a URL"""
        features = {}
        parsed_url = urlparse(url)
        
        # Basic features
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed_url.netloc)
        features['path_length'] = len(parsed_url.path)
        
        # Security indicators
        features['has_https'] = int(parsed_url.scheme == 'https')
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
        features['num_digits'] = len(re.findall(r'\d', url))
        
        # Suspicious patterns
        features['has_ip_address'] = int(bool(re.match(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            parsed_url.netloc
        )))
        features['num_subdomains'] = len(parsed_url.netloc.split('.')) - 1
        
        # Suspicious words check
        suspicious_words = ['login', 'verify', 'account', 'secure', 'banking']
        features['has_suspicious_words'] = int(any(word in url.lower() for word in suspicious_words))
        
        return np.array([features[feature] for feature in self.feature_names])
    
    def fit_transform(self, urls):
        """Extract and scale features for multiple URLs"""
        features = np.array([self.extract_features(url) for url in urls])
        return self.scaler.fit_transform(features)
    
    def transform(self, urls):
        """Transform new URLs using fitted scaler"""
        features = np.array([self.extract_features(url) for url in urls])
        return self.scaler.transform(features)