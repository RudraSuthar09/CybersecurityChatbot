import requests
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import zipfile

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
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

def download_cicids_full():
    base_dir = 'datasets/intrusion_detection/cicids/2017'
    create_directory(base_dir)
    
    # Updated CIC-IDS-2017 dataset files with direct download links
    files = {
        'Monday-WorkingHours.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': 'https://iscx.ca/download/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    }
    
    print("\n📥 Downloading Full CICIDS2017 Dataset from CIC")
    print("=" * 50)
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: RudraSuthar09")
    print("This will download approximately 6.4GB of data...")
    
    successful_downloads = 0
    total_files = len(files)
    
    for filename, url in files.items():
        if download_file(url, filename, base_dir):
            successful_downloads += 1
            
            # Verify the file is a valid CSV
            try:
                df = pd.read_csv(os.path.join(base_dir, filename), nrows=5)
                print(f"✅ Verified {filename} as valid CSV")
            except Exception as e:
                print(f"⚠️ Warning: {filename} may not be a valid CSV: {str(e)}")
    
    print("\n📊 Download Summary:")
    print(f"Total files attempted: {total_files}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {total_files - successful_downloads}")
    
    if successful_downloads == total_files:
        print("\n✅ Successfully downloaded complete CICIDS2017 dataset!")
    else:
        print("\n⚠️ Some files failed to download. Please try the alternative method.")
        print("\nAlternative download instructions:")
        print("1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("2. Fill out the registration form")
        print("3. Download the dataset files directly from the website")

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    expected_dirs = ['datasets/intrusion_detection/cicids/2017']
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✅ Found: {dir_path}")
            print(f"📁 Contains {len(files)} files:")
            total_size = 0
            for file in files:
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    total_size += size_mb
                    print(f"   - {file} ({size_mb:.2f} MB)")
            print(f"\nTotal size: {total_size:.2f} MB")

if __name__ == "__main__":
    print(f"Starting Full CICIDS2017 Dataset download from CIC")
    print(f"Current Time: 2025-03-30 17:25:28 UTC")
    print(f"User: RudraSuthar09")
    
    print("\nWarning: This will download the complete CICIDS2017 dataset (~6.4GB).")
    print("Make sure you have:")
    print("1. At least 7GB of free disk space")
    print("2. Stable internet connection")
    print("3. Permission to write to the current directory")
    
    print("\nImportant: If automatic download fails, you'll need to:")
    print("1. Register at: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("2. Download files manually")
    
    response = input("Do you want to proceed with automatic download? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        download_cicids_full()
        verify_structure()
    else:
        print("Download cancelled by user.")