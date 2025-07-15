"""
Auto-install required packages for Parkinson Classification
"""

import subprocess
import sys
import pkg_resources
from packaging import version

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check and install required packages"""
    
    # Core requirements with version checking
    requirements = {
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scipy': '1.7.0',
        'scikit-learn': '1.0.0',
        'matplotlib': '3.4.0',
        'nibabel': '3.2.0',
        'openpyxl': '3.0.7',
        'plotly': '5.0.0'
    }
    
    print("🔍 Checking required packages...")
    
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in requirements.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if version.parse(installed_version) < version.parse(min_version):
                outdated_packages.append(f"{package}>={min_version}")
            else:
                print(f"✅ {package} {installed_version}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(f"{package}>={min_version}")
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    
    # Update outdated packages
    if outdated_packages:
        print(f"\n🔄 Updating outdated packages: {outdated_packages}")
        for package in outdated_packages:
            print(f"Updating {package}...")
            if install_package(package):
                print(f"✅ {package} updated successfully")
            else:
                print(f"❌ Failed to update {package}")
    
    # Optional packages
    optional_packages = ['torch', 'statsmodels']
    print(f"\n🔧 Checking optional packages...")
    
    for package in optional_packages:
        try:
            installed_version = pkg_resources.get_distribution(package).version
            print(f"✅ {package} {installed_version} (optional)")
        except pkg_resources.DistributionNotFound:
            print(f"⚠️  {package} not installed (optional)")
    
    print(f"\n✨ Package check completed!")

if __name__ == "__main__":
    check_and_install_packages()