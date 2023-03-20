# system_info.py
import torch
import platform
import psutil
import GPUtil

def check_gpu_availability():
    return torch.cuda.is_available()

def get_system_info():
    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "CPU": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    return system_info

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info = []
    
    for gpu in gpus:
        info = {
            "ID": gpu.id,
            "Name": gpu.name,
            "Load (%)": gpu.load * 100,
            "Memory Used (GB)": round(gpu.memoryUsed / 1024, 2),
            "Memory Total (GB)": round(gpu.memoryTotal / 1024, 2),
        }
        gpu_info.append(info)
        
    return gpu_info

def main():
    print("\nSystem Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    if check_gpu_availability():
        print("\nGPU(s) Information:")
        gpu_info = get_gpu_info()
        for info in gpu_info:
            for key, value in info.items():
                print(f"{key}: {value}")
            print()
        
        # Set up the PyTorch framework to use the GPU
        device = torch.device("cuda:0")
        print("PyTorch is configured to use the GPU.")
        
    else:
        print("\nNo GPU available. PyTorch will use the CPU.")

if __name__ == "__main__":
    main()
