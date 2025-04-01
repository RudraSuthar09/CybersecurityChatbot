import os
import json
from datetime import datetime

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_nist_data(output_dir):
    nist_framework = {
        "framework_version": "1.1",
        "last_updated": "2025-03-30",
        "generated_by": "RudraSuthar09",
        "generated_at": "2025-03-30 17:06:28",
        "core_functions": {
            "IDENTIFY": {
                "description": "Develop organizational understanding to manage cybersecurity risk to systems, people, assets, data, and capabilities.",
                "categories": [
                    "Asset Management",
                    "Business Environment",
                    "Governance",
                    "Risk Assessment",
                    "Risk Management Strategy"
                ]
            },
            "PROTECT": {
                "description": "Develop and implement appropriate safeguards to ensure delivery of critical services.",
                "categories": [
                    "Identity Management and Access Control",
                    "Awareness and Training",
                    "Data Security",
                    "Information Protection Processes",
                    "Protective Technology"
                ]
            },
            "DETECT": {
                "description": "Develop and implement appropriate activities to identify the occurrence of a cybersecurity event.",
                "categories": [
                    "Anomalies and Events",
                    "Security Continuous Monitoring",
                    "Detection Processes"
                ]
            },
            "RESPOND": {
                "description": "Develop and implement appropriate activities to take action regarding a detected cybersecurity incident.",
                "categories": [
                    "Response Planning",
                    "Communications",
                    "Analysis",
                    "Mitigation",
                    "Improvements"
                ]
            },
            "RECOVER": {
                "description": "Develop and implement appropriate activities to maintain plans for resilience and to restore any capabilities or services that were impaired due to a cybersecurity incident.",
                "categories": [
                    "Recovery Planning",
                    "Improvements",
                    "Communications"
                ]
            }
        },
        "implementation_tiers": [
            {
                "tier": "Tier 1",
                "name": "Partial",
                "description": "Risk management practices are not formalized, and risk is managed in an ad hoc manner."
            },
            {
                "tier": "Tier 2",
                "name": "Risk Informed",
                "description": "Risk management practices are approved by management but may not be established organization-wide."
            },
            {
                "tier": "Tier 3",
                "name": "Repeatable",
                "description": "Organization-wide risk management practices are formally approved and expressed as policy."
            },
            {
                "tier": "Tier 4",
                "name": "Adaptive",
                "description": "Organization adapts cybersecurity practices based on previous and current cybersecurity activities."
            }
        ]
    }
    
    # Save NIST Framework data as JSON
    nist_file = os.path.join(output_dir, 'nist_cybersecurity_framework.json')
    with open(nist_file, 'w') as f:
        json.dump(nist_framework, f, indent=2)
    print(f"\n✅ Created NIST Framework data in {nist_file}")
    
    # Create NIST Framework markdown guidelines
    nist_md = """# NIST Cybersecurity Framework

## Core Functions

### 1. IDENTIFY
Develop organizational understanding to manage cybersecurity risk to systems, assets, data, and capabilities.
- Asset Management
- Business Environment
- Governance
- Risk Assessment
- Risk Management Strategy

### 2. PROTECT
Develop and implement appropriate safeguards to ensure delivery of critical services.
- Identity Management and Access Control
- Awareness and Training
- Data Security
- Information Protection Processes
- Protective Technology

### 3. DETECT
Develop and implement appropriate activities to identify cybersecurity events.
- Anomalies and Events
- Security Continuous Monitoring
- Detection Processes

### 4. RESPOND
Develop and implement appropriate activities to take action regarding detected events.
- Response Planning
- Communications
- Analysis
- Mitigation
- Improvements

### 5. RECOVER
Develop and implement appropriate activities to maintain plans for resilience.
- Recovery Planning
- Improvements
- Communications

## Implementation Tiers

1. **Tier 1 - Partial**
   - Risk management practices are not formalized
   - Risk is managed in an ad hoc manner

2. **Tier 2 - Risk Informed**
   - Risk management practices approved by management
   - May not be established organization-wide

3. **Tier 3 - Repeatable**
   - Organization-wide risk management practices
   - Formally approved and expressed as policy

4. **Tier 4 - Adaptive**
   - Organization adapts cybersecurity practices
   - Based on previous and current activities

_Generated by: RudraSuthar09_
_Last Updated: 2025-03-30_
"""
    
    guidelines_file = os.path.join(output_dir, 'nist_framework_guidelines.md')
    with open(guidelines_file, 'w') as f:
        f.write(nist_md)
    print(f"✅ Created NIST Framework guidelines in {guidelines_file}")
    
    return True

def verify_structure():
    print("\n🔍 Verifying directory structure...")
    expected_dirs = ['datasets/security/nist']
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")

if __name__ == "__main__":
    print(f"Starting NIST Framework Data creation at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    base_dir = 'datasets/security/nist'
    create_directory(base_dir)
    
    print("\nThis script will create NIST Cybersecurity Framework dataset.")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        create_nist_data(base_dir)
        verify_structure()
    else:
        print("Operation cancelled by user.")