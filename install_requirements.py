import subprocess
import sys
import time

def install_requirements():
    print("Installing required packages...")
    print(f"Current Date and Time (UTC): 2025-03-31 04:49:39")
    print(f"Current User: RudraSuthar09")
    
    # List of required packages
    requirements = [
        'pyyaml',        # For YAML file handling
        'torch',         # PyTorch for deep learning
        'numpy',         # For numerical operations
        'pandas',        # For data manipulation
        'scikit-learn', # For machine learning utilities
        'matplotlib',    # For plotting
        'seaborn'       # For advanced plotting
    ]
    
    # Install each package
    for package in requirements:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            continue

    print("\nAll required packages have been installed!")

if __name__ == "__main__":
    install_requirements()