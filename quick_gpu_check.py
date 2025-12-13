#!/usr/bin/env python3

import GPUtil
import psutil
import time
import argparse
from datetime import datetime

def get_quick_stats():
    """Get quick system statistics"""
    
    print("="*50)
    print("ISAAC LAB QUICK GPU & SYSTEM CHECK")
    print("="*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU Information
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"\nGPU {i}: {gpu.name}")
                print(f"  Usage: {gpu.load * 100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%)")
                print(f"  Temperature: {gpu.temperature}°C")
        else:
            print("\nNo NVIDIA GPUs detected")
    except Exception as e:
        print(f"\nGPU Error: {e}")
    
    # CPU Information  
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"\nCPU: {cpu_count} cores")
    print(f"  Usage: {cpu_percent:.1f}%")
    
    # RAM Information
    memory = psutil.virtual_memory()
    print(f"\nRAM: {memory.total/(1024**3):.1f}GB total")
    print(f"  Used: {memory.used/(1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"  Available: {memory.available/(1024**3):.1f}GB")
    
    # Disk Information
    disk = psutil.disk_usage('/')
    print(f"\nDisk: {disk.total/(1024**3):.1f}GB total")
    print(f"  Used: {disk.used/(1024**3):.1f}GB ({(disk.used/disk.total)*100:.1f}%)")
    print(f"  Free: {disk.free/(1024**3):.1f}GB")
    
    print("="*50)

def monitor_continuously(interval=5, duration=None):
    """Monitor system continuously"""
    start_time = time.time()
    
    try:
        while True:
            get_quick_stats()
            
            if duration and (time.time() - start_time) >= duration:
                break
                
            print(f"\nRefreshing in {interval} seconds... (Press Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Quick GPU and System Check')
    parser.add_argument('--continuous', '-c', action='store_true',
                      help='Monitor continuously')
    parser.add_argument('--interval', '-i', type=int, default=5,
                      help='Update interval in seconds (for continuous mode)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                      help='Total monitoring duration in seconds')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously(args.interval, args.duration)
    else:
        get_quick_stats()

if __name__ == "__main__":
    main()