import requests
import os
import pandas as pd
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
        
        response = requests.get(url, stream=True, headers=headers, verify=True)
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

def process_dataset(input_file, output_file):
    try:
        # Read the raw data
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        processed_lines = []
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type', 'level'
        ]
        
        processed_lines.append(','.join(columns))
        
        for line in lines:
            processed_lines.append(line.strip())
        
        # Write processed data
        with open(output_file, 'w') as f:
            f.write('\n'.join(processed_lines))
        
        # Read the processed CSV for statistics
        df = pd.read_csv(output_file)
        print(f"✅ Successfully processed and saved to {output_file}")
        
        # Print dataset statistics
        print("\n📊 Dataset Statistics:")
        print(f"Total records: {len(df):,}")
        print("\nAttack type distribution:")
        print(df['attack_type'].value_counts().head())
        
        return True
    except Exception as e:
        print(f"❌ Error processing dataset: {str(e)}")
        return False

def download_nslkdd():
    base_dir = 'datasets/intrusion_detection/nslkdd'
    raw_dir = os.path.join(base_dir, 'raw')
    processed_dir = os.path.join(base_dir, 'processed')
    
    create_directory(raw_dir)
    create_directory(processed_dir)
    
    # NSL-KDD dataset files
    files = {
        'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
        'KDDTest+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
    }
    
    print("\n📥 Downloading NSL-KDD Dataset")
    print("=" * 50)
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: RudraSuthar09")
    print("This will download and process the NSL-KDD dataset...")
    
    successful_downloads = 0
    
    for filename, url in files.items():
        raw_file = os.path.join(raw_dir, filename)
        processed_file = os.path.join(processed_dir, filename.replace('.txt', '.csv'))
        
        if download_file(url, filename, raw_dir):
            successful_downloads += 1
            process_dataset(raw_file, processed_file)
    
    if successful_downloads == len(files):
        print("\n✅ Successfully downloaded and processed NSL-KDD dataset!")
        
        # Create dataset info file
        info_file = os.path.join(base_dir, 'dataset_info.json')
        dataset_info = {
            "name": "NSL-KDD",
            "description": "Improved version of the KDD'99 dataset for network intrusion detection",
            "downloaded_by": "RudraSuthar09",
            "downloaded_at": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            "files": list(files.keys()),
            "features": 41,
            "attack_types": [
                "normal", "DoS", "Probe", "R2L", "U2R"
            ]
        }
        
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Created dataset information file: {info_file}")
    else:
        print("\n⚠️ Some files failed to download. Please try again.")

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    for dir_path in ['datasets/intrusion_detection/nslkdd/raw', 
                     'datasets/intrusion_detection/nslkdd/processed']:
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
    print(f"Starting NSL-KDD Dataset download")
    print(f"Current Time: 2025-03-30 17:34:24")
    print(f"User: RudraSuthar09")
    
    print("\nThis script will:")
    print("1. Download the NSL-KDD dataset (safer alternative to CICIDS2017)")
    print("2. Process the data into a clean CSV format")
    print("3. Create documentation and statistics")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        download_nslkdd()
        verify_structure()
    else:
        print("Download cancelled by user.")