import subprocess
import platform
import torch

def get_cuda_version():
    try:
        # 使用torch来获取CUDA版本
        cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        return f"无法获取CUDA版本: {e}"

def get_python_version():
    # 使用platform库获取Python版本
    python_version = platform.python_version()
    return python_version

def get_installed_packages():
    try:
        # 使用pip命令获取已安装的库及其版本
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"无法获取已安装的库: {e}"

def get_edition_all():
    print("CUDA版本:", get_cuda_version())
    print("Python版本:", get_python_version())
    print("已安装的库及其版本:")
    print(get_installed_packages())