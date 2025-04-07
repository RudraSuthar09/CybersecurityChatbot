import numpy as np
import re
import math

def calculate_entropy(text):
    """Calculates the Shannon entropy of a string."""
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def translate_to_features(text: str):
    text = text.lower()

    # Normal and suspicious keywords
    normal_keywords = [
        "hello", "hi", "how are you", "check email", "browse", "safe", "normal",
        "news", "weather", "search", "google", "benign", "open site", "web access"
    ]

    # Group suspicious keywords by category
    suspicious_categories = {
        "DoS": ["neptune", "smurf", "teardrop", "land", "ping of death"],
        "Probe": ["nmap", "portsweep", "satan", "ipsweep", "scan", "probe"],
        "R2L": ["guess password", "ftp", "imap", "telnet", "phishing", "multihop"],
        "U2R": ["buffer overflow", "rootkit", "perl", "loadmodule", "xterm", "injection"],
        "Malware": ["malware", "virus", "trojan", "ransomware", "keylogger", "ddos", "backdoor"]
    }

    features = []

    # 1. Character length
    features.append(len(text))

    # 2. Word count
    words = text.split()
    features.append(len(words))

    # 3. Count of suspicious keyword matches
    suspicious_count = sum(1 for cat in suspicious_categories.values() for word in cat if word in text)
    features.append(suspicious_count)

    # 4. Count of normal keywords
    normal_count = sum(1 for word in normal_keywords if word in text)
    features.append(normal_count)

    # 5. Ratio suspicious/words
    ratio = suspicious_count / (len(words) + 1e-5)
    features.append(ratio)

    # 6. Has numbers
    features.append(1 if any(char.isdigit() for char in text) else 0)

    # 7. Punctuation count
    features.append(len(re.findall(r'[^\w\s]', text)))

    # 8. Is question
    features.append(1 if "?" in text else 0)

    # 9–13. Category-wise keyword detection
    for category in ["DoS", "Probe", "R2L", "U2R", "Malware"]:
        match_count = sum(1 for word in suspicious_categories[category] if word in text)
        features.append(match_count)

    # 14. Contains IP address-like pattern
    features.append(1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text) else 0)

    # 15. Contains URL
    features.append(1 if re.search(r'(https?://\S+|www\.\S+)', text) else 0)

    # 16. Contains file extensions
    features.append(1 if re.search(r'\.(exe|bat|sh|py|php|js|jar)', text) else 0)

    # 17. Average word length
    avg_word_len = sum(len(w) for w in words) / (len(words) + 1e-5)
    features.append(avg_word_len)

    # 18. Entropy (randomness of text)
    features.append(calculate_entropy(text))

    # 🔧 Pad to 42 features
    MAX_FEATURES = 42
    while len(features) < MAX_FEATURES:
        features.append(0.0)

    # Trim if it exceeds
    features = features[:MAX_FEATURES]

    return np.array(features).tolist()
