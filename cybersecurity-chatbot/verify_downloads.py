import json
import os
import pandas as pd
from datetime import datetime

def verify_nslkdd_data():
    print("\n=== NSL-KDD Dataset Verification ===")
    
    # Check directory structure
    base_dir = 'datasets/intrusion_detection/nslkdd'
    required_dirs = ['raw', 'processed']
    required_files = {
        'raw': ['KDDTrain+.txt', 'KDDTest+.txt'],
        'processed': ['KDDTrain+.csv', 'KDDTest+.csv']
    }
    
    if not os.path.exists(base_dir):
        print("❌ Error: NSL-KDD base directory not found!")
        return False
        
    # Verify directories and files
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Error: {dir_name} directory not found!")
            return False
            
        # Check required files
        for filename in required_files[dir_name]:
            file_path = os.path.join(dir_path, filename)
            if not os.path.exists(file_path):
                print(f"❌ Error: {filename} not found in {dir_name}!")
                return False
                
            # Get file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"\n✅ Found {filename}")
            print(f"   Size: {file_size:.2f} MB")
            
            # Check last modified date
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   Last modified: {mod_date}")
            
            # For processed CSV files, verify content
            if dir_name == 'processed' and filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    print(f"   Records: {len(df):,}")
                    print(f"   Features: {len(df.columns)}")
                    print("   Attack type distribution:")
                    attack_dist = df['attack_type'].value_counts()
                    for attack_type, count in attack_dist.head().items():
                        print(f"   - {attack_type}: {count:,}")
                except Exception as e:
                    print(f"❌ Error reading CSV file: {e}")
                    return False
    
    return True

def verify_mitre_data():
    print("\n=== MITRE ATT&CK Dataset Verification ===")
    
    # Check if file exists
    mitre_file = 'datasets/threat_detection/mitre/enterprise-attack.json'
    if not os.path.exists(mitre_file):
        print("❌ Error: MITRE ATT&CK dataset file not found!")
        return False
    
    try:
        # Try to read and parse the JSON file
        with open(mitre_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation checks
        if not isinstance(data, dict):
            print("❌ Error: Invalid data format!")
            return False
            
        objects = data.get('objects', [])
        if not objects:
            print("❌ Error: No objects found in the dataset!")
            return False
        
        # Count different types of objects
        object_types = {}
        for obj in objects:
            obj_type = obj.get('type', 'unknown')
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Print verification results
        print("✅ File exists and is valid JSON")
        print(f"✅ Total objects found: {len(objects)}")
        print("\nObject types found:")
        for obj_type, count in object_types.items():
            print(f"- {obj_type}: {count}")
        
        # Check file size
        file_size = os.path.getsize(mitre_file) / (1024 * 1024)  # Convert to MB
        print(f"\nFile size: {file_size:.2f} MB")
        
        # Check last modified date
        mod_time = os.path.getmtime(mitre_file)
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Last modified: {mod_date}")
        
        return True
        
    except json.JSONDecodeError:
        print("❌ Error: File contains invalid JSON!")
        return False
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred: {e}")
        return False

def main():
    print(f"Starting verification process...")
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: RudraSuthar09")
    
    # Verify NSL-KDD dataset
    nslkdd_success = verify_nslkdd_data()
    
    # Verify MITRE dataset
    mitre_success = verify_mitre_data()
    
    # Print overall status
    print("\n=== Overall Verification Status ===")
    print(f"NSL-KDD Dataset: {'✅ Success' if nslkdd_success else '❌ Failed'}")
    print(f"MITRE Dataset: {'✅ Success' if mitre_success else '❌ Failed'}")
    
    if nslkdd_success and mitre_success:
        print("\n✅ All verifications completed successfully!")
    else:
        print("\n❌ Some verifications failed!")

if __name__ == "__main__":
    main()