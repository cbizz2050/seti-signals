# main.py
import os
from train import train_models
from system_info import get_system_info, check_gpu_availability, get_gpu_info
import torch

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
        
    
    # Create and train models
    train_models()

if __name__ == "__main__":
    main()
