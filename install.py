import launch
import os
import subprocess

requirements = [
"diffusers==0.25.0",
"safetensors",
"dadaptation",
"prodigyopt",
"wandb",
"lycoris_lora",
"pandas",
"matplotlib"
]

for module in requirements:
    if not launch.is_installed(module):
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

    if not launch.is_installed(package_name):
        launch.run_pip(f"install { 'git+' + package_url if 'http' in package_url else package_url}", package_name)
    else:
        print(f"{package_name} is already installed.")
