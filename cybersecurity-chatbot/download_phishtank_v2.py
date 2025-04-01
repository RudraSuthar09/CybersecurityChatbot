import requests
import os
import json
from datetime import datetime
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def download_phishtank_data(output_dir):
    # PhishTank Developer API endpoint
    url = 'http://data.phishtank.com/data/online-valid.json'
    
    # Headers to identify our application
    headers = {
        'User-Agent': 'PhishTankResearchBot/1.0',
        'Accept': 'application/json'
    }
    
    output_file = os.path.join(output_dir, 'phishtank_data.json')
    
    try:
        print("\nDownloading PhishTank data...")
        response = requests.get(url, headers=headers, stream=True)
        
        # If we get a 403, provide instructions
        if response.status_code == 403:
            print("\n❌ Access Denied. Let's use the alternative download method:")
            print("\n1. Visit: https://www.phishtank.com/developer_info.php")
            print("2. Register for a free API key")
            print("3. Download the database directly from the developer portal")
            return False
            
        response.raise_for_status()
        
        # Save the response to a file
        with open(output_file, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Successfully downloaded PhishTank data to {output_file}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nAlternative download method:")
        print("1. Visit: https://www.phishtank.com/developer_info.php")
        print("2. Register for a free API key")
        print("3. Download the database directly from the developer portal")
        return False

def create_sample_data(output_dir):
    """Create sample phishing data for testing"""
    sample_data = {
        "generated": str(datetime.utcnow()),
        "sample_phishing_urls": [
            {
                "url": "http://example.phishing.com",
                "submission_time": "2025-03-30T17:01:52+00:00",
                "verified": True,
                "verification_time": "2025-03-30T17:05:00+00:00",
                "target": "Sample Bank"
            },
            {
                "url": "https://fake.login.com",
                "submission_time": "2025-03-30T16:55:00+00:00",
                "verified": True,
                "verification_time": "2025-03-30T17:00:00+00:00",
                "target": "Social Media"
            }
        ]
    }
    
    sample_file = os.path.join(output_dir, 'sample_phishing_data.json')
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"\n✅ Created sample phishing data in {sample_file}")
    return True

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    expected_dirs = ['datasets/phishing/phishtank']
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")

if __name__ == "__main__":
    print(f"Starting PhishTank Database download at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    base_dir = 'datasets/phishing/phishtank'
    create_directory(base_dir)
    
    print("\nNote: This script will attempt to download PhishTank data and create a sample dataset.")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = download_phishtank_data(base_dir)
        if not success:
            print("\nCreating sample dataset instead...")
            create_sample_data(base_dir)
        
        verify_structure()
    else:
        print("Download cancelled by user.")