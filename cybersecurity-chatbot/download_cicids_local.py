import os
import shutil
from datetime import datetime

def setup_directories():
    base_dir = 'datasets/intrusion_detection/cicids/2017'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    return base_dir

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    expected_dirs = [
        'datasets/threat_detection/mitre',
        'datasets/intrusion_detection/cicids/2017'
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")

def print_instructions():
    print("""
=== CICIDS 2017 Dataset Download Instructions ===

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html

2. Download these files:
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

3. Move the downloaded files to:
   D:\\Chatbot datasets\\cybersecurity-chatbot\\datasets\\intrusion_detection\\cicids\\2017\\

4. Run this script again to verify the files.

Note: The dataset is large (approximately 50GB total).
Make sure you have enough disk space!
""")

def check_downloaded_files(base_dir):
    expected_files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    ]
    
    print("\n🔍 Checking for downloaded files:")
    found_files = []
    missing_files = []
    
    for file in expected_files:
        if os.path.exists(os.path.join(base_dir, file)):
            found_files.append(file)
            size_mb = os.path.getsize(os.path.join(base_dir, file)) / (1024 * 1024)
            print(f"✅ Found: {file} ({size_mb:.2f} MB)")
        else:
            missing_files.append(file)
            print(f"❌ Missing: {file}")
    
    return found_files, missing_files

if __name__ == "__main__":
    print(f"CICIDS Dataset Helper - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    base_dir = setup_directories()
    verify_structure()
    
    print_instructions()
    
    response = input("\nWould you like to check for downloaded files? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        found_files, missing_files = check_downloaded_files(base_dir)
        
        print(f"\n📊 Summary:")
        print(f"Found files: {len(found_files)}")
        print(f"Missing files: {len(missing_files)}")
        
        if missing_files:
            print("\nPlease download the missing files and run this script again.")
    else:
        print("File check cancelled.")