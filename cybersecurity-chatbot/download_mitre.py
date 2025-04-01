import requests
import json
import os

# Create directory if it doesn't exist
os.makedirs('datasets/threat_detection/mitre', exist_ok=True)

# Download the MITRE ATT&CK data
print("Downloading MITRE ATT&CK data...")
url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

try:
    # Download the data
    response = requests.get(url)
    response.raise_for_status()  # Check if download was successful
    
    # Save the data
    output_file = 'datasets/threat_detection/mitre/enterprise-attack.json'
    with open(output_file, 'w') as f:
        json.dump(response.json(), f, indent=2)
    
    print(f"Successfully downloaded MITRE ATT&CK data to {output_file}")
    
    # Print basic statistics
    data = response.json()
    print(f"\nTotal number of objects: {len(data.get('objects', []))}")
    
except requests.RequestException as e:
    print(f"Error downloading data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print("\nDone!")