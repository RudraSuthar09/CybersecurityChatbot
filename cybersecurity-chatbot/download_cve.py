import requests
import os
import json
from datetime import datetime
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def download_file(url, filename, output_dir):
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"⏭️ File already exists: {filename}")
        return True
        
    try:
        print(f"\nDownloading {filename}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                pbar.update(size)
                
        print(f"✅ Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def download_cve():
    base_dir = 'datasets/vulnerabilities/cve'
    create_directory(base_dir)
    
    # Recent CVE data files from NVD
    current_year = datetime.utcnow().year
    years = list(range(current_year - 2, current_year + 1))  # Last 2 years + current year
    
    cve_files = {
        f'nvdcve-1.1-{year}.json.gz': 
        f'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz'
        for year in years
    }

    print("\n📥 Downloading CVE Database")
    print("=" * 50)
    
    successful_downloads = 0
    total_files = len(cve_files)
    
    for filename, url in cve_files.items():
        if download_file(url, filename, base_dir):
            successful_downloads += 1
            
    print("\n📊 Download Summary:")
    print(f"Total files attempted: {total_files}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {total_files - successful_downloads}")

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    
    expected_dirs = [
        'datasets/vulnerabilities/cve'
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")

if __name__ == "__main__":
    print(f"Starting CVE Database download at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("Current directory structure check:")
    verify_structure()
    
    response = input("\nDo you want to proceed with downloading CVE Database? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        download_cve()
        print("\nVerifying final structure:")
        verify_structure()
    else:
        print("Download cancelled by user.")