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

def create_owasp_data(output_dir):
    owasp_top_10 = {
        "OWASP_Top_10_2021": [
            {
                "rank": "A01",
                "category": "Broken Access Control",
                "description": "Moving up from the fifth position, 94% of applications were tested for some form of broken access control.",
                "prevention": [
                    "Implement proper access controls",
                    "Deny by default",
                    "Implement rate limiting"
                ]
            },
            {
                "rank": "A02",
                "category": "Cryptographic Failures",
                "description": "Previously known as Sensitive Data Exposure, failures related to cryptography often lead to sensitive data exposure or system compromise.",
                "prevention": [
                    "Encrypt all sensitive data at rest",
                    "Use strong encryption algorithms",
                    "Proper key management"
                ]
            },
            {
                "rank": "A03",
                "category": "Injection",
                "description": "Injection flaws, such as SQL injection, NoSQL injection, and LDAP injection, occur when untrusted data is sent to an interpreter.",
                "prevention": [
                    "Use parameterized queries",
                    "Input validation",
                    "Escape special characters"
                ]
            }
        ],
        "security_guidelines": {
            "authentication": [
                "Implement MFA where possible",
                "Use secure password hashing",
                "Implement proper session management"
            ],
            "authorization": [
                "Implement role-based access control",
                "Use principle of least privilege",
                "Regular access reviews"
            ],
            "data_protection": [
                "Encrypt sensitive data in transit and at rest",
                "Implement proper key management",
                "Regular security assessments"
            ]
        },
        "generated": str(datetime.utcnow()),
        "last_updated": "2025-03-30"
    }
    
    # Save OWASP data
    owasp_file = os.path.join(output_dir, 'owasp_security_data.json')
    with open(owasp_file, 'w') as f:
        json.dump(owasp_top_10, f, indent=2)
    print(f"\n✅ Created OWASP security data in {owasp_file}")
    
    # Create OWASP guidelines markdown
    guidelines_md = """# OWASP Security Guidelines

## Top 10 Security Risks (2021)

1. **Broken Access Control**
   - Implementation of proper access controls
   - Deny access by default
   - Regular access review

2. **Cryptographic Failures**
   - Use strong encryption
   - Proper key management
   - Secure data at rest and in transit

3. **Injection**
   - Use parameterized queries
   - Input validation
   - Output encoding

## Best Practices

### Authentication
- Implement Multi-Factor Authentication
- Use secure password storage
- Session management

### Authorization
- Role-Based Access Control (RBAC)
- Principle of Least Privilege
- Regular access reviews

### Data Protection
- Encryption in transit and at rest
- Key management
- Regular security assessments

_Last Updated: 2025-03-30_
"""
    
    guidelines_file = os.path.join(output_dir, 'security_guidelines.md')
    with open(guidelines_file, 'w') as f:
        f.write(guidelines_md)
    print(f"✅ Created security guidelines in {guidelines_file}")
    
    return True

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    expected_dirs = ['datasets/security/owasp']
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")

if __name__ == "__main__":
    print(f"Starting OWASP Security Data creation at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    base_dir = 'datasets/security/owasp'
    create_directory(base_dir)
    
    print("\nThis script will create OWASP Top 10 and security guidelines dataset.")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        create_owasp_data(base_dir)
        verify_structure()
    else:
        print("Operation cancelled by user.")