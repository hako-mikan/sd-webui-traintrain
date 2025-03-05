import launch
import os
import importlib.metadata
import importlib.util
from packaging.version import Version
from packaging.requirements import Requirement

requirements = [
"diffusers",
"safetensors",
"dadaptation==3.2",
"prodigyopt",
"lycoris_lora",
"pandas",
"matplotlib",
"schedulefree",
"pytorch-optimizer",
]

def is_installed(pip_package):
    """
    Check if a package is installed and meets version requirements specified in pip-style format.

    Args:
        pip_package (str): Package name in pip-style format (e.g., "numpy>=1.22.0").
    
    Returns:
        bool: True if the package is installed and meets the version requirement, False otherwise.
    """
    try:
        # Parse the pip-style package name and version constraints
        requirement = Requirement(pip_package)
        package_name = requirement.name
        specifier = requirement.specifier  # e.g., >=1.22.0
        
        # Check if the package is installed
        dist = importlib.metadata.distribution(package_name)
        installed_version = Version(dist.version)
        
        # Check version constraints
        if specifier.contains(installed_version):
            return True
        else:
            print(f"Installed version of {package_name} ({installed_version}) does not satisfy the requirement ({specifier}).")
            return False
    except importlib.metadata.PackageNotFoundError:
        print(f"Package {pip_package} is not installed.")
        return False
    

for module in requirements:
    if not is_installed(module):
        launch.run_pip(f"install {module}", module)

try:
    from ldm_patched.modules import model_management
except:
    if os.name == 'nt':
        package_url = "bitsandbytes>=0.43.0"
        package_name = "bitsandbytes>=0.43.0" 
    else:
        package_url = "bitsandbytes"
        package_name = "bitsandbytes"

    if not is_installed(package_name):
        launch.run_pip(f"install { 'git+' + package_url if 'http' in package_url else package_url}", package_name)
    else:
        print(f"{package_name} is already installed.")
