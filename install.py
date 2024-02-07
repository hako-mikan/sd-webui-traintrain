import launch
import os
import subprocess

requirements = [
"scipy",
"diffusers==0.20.0",
"safetensors",
"pyyaml",
"pydantic",
"dadaptation",
"lion-pytorch",
"prodigyopt",
"wandb",
"omegaconf",
"invisible-watermark==0.2.0",
"lycoris_lora",
"toml",
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
        package_url = "https://github.com/Keith-Hon/bitsandbytes-windows.git"
        package_name = "bitsandbytes" 
    else:
        package_url = "bitsandbytes"
        package_name = "bitsandbytes"

    if not launch.is_installed(package_name):
        launch.run_pip(f"install { 'git+' + package_url if 'http' in package_url else package_url}", package_name)
    else:
        print(f"{package_name} is already installed.")
