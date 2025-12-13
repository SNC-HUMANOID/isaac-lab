#!/usr/bin/env python3

import psutil
import GPUtil
import time
import json
import argparse
import threading
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import subprocess

class TrainingAnalyzer:
    def __init__(self, update_interval=1.0, history_size=300):
        self.update_interval = update_interval
        self.history_size = history_size
        
        # Data storage
        self.gpu_usage_history = deque(maxlen=history_size)
        self.gpu_memory_history = deque(maxlen=history_size)
        self.gpu_temp_history = deque(maxlen=history_size)
        self.cpu_usage_history = deque(maxlen=history_size)
        self.ram_usage_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # Training metrics
        self.training_start_time = None
        self.episode_count = 0
        self.total_rewards = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
    def get_gpu_stats(self):
        """Get GPU utilization, memory usage, and temperature"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'usage': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'name': gpu.name
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        return None
    
    def get_cpu_stats(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=None)
    
    def get_ram_stats(self):
        """Get RAM usage statistics"""
        memory = psutil.virtual_memory()
        return {
            'used': memory.used / (1024**3),  # GB
            'total': memory.total / (1024**3),  # GB
            'percent': memory.percent
        }
    
    def monitor_system(self):
        """Monitor system resources in a separate thread"""
        while self.is_monitoring:
            timestamp = time.time()
            
            # Get GPU stats
            gpu_stats = self.get_gpu_stats()
            if gpu_stats:
                self.gpu_usage_history.append(gpu_stats['usage'])
                self.gpu_memory_history.append(gpu_stats['memory_percent'])
                self.gpu_temp_history.append(gpu_stats['temperature'])
            else:
                self.gpu_usage_history.append(0)
                self.gpu_memory_history.append(0)
                self.gpu_temp_history.append(0)
            
            # Get CPU and RAM stats
            cpu_usage = self.get_cpu_stats()
            ram_stats = self.get_ram_stats()
            
            self.cpu_usage_history.append(cpu_usage)
            self.ram_usage_history.append(ram_stats['percent'])
            self.timestamps.append(timestamp)
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.training_start_time = time.time()
            self.monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
            self.monitor_thread.start()
            print("Started system monitoring...")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Stopped system monitoring...")
    
    def get_current_stats(self):
        """Get current system statistics"""
        gpu_stats = self.get_gpu_stats()
        cpu_usage = self.get_cpu_stats()
        ram_stats = self.get_ram_stats()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': time.time() - self.training_start_time if self.training_start_time else 0,
            'cpu_usage': cpu_usage,
            'ram_usage': ram_stats,
            'gpu_stats': gpu_stats
        }
        
        return stats
    
    def print_current_stats(self):
        """Print current statistics to console"""
        stats = self.get_current_stats()
        
        print("\n" + "="*60)
        print("ISAAC LAB TRAINING ANALYSIS")
        print("="*60)
        
        if stats['training_duration'] > 0:
            duration = stats['training_duration']
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            print(f"Training Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # CPU Stats
        print(f"\nCPU Usage: {stats['cpu_usage']:.1f}%")
        
        # RAM Stats
        ram = stats['ram_usage']
        print(f"RAM Usage: {ram['used']:.1f}GB / {ram['total']:.1f}GB ({ram['percent']:.1f}%)")
        
        # GPU Stats
        if stats['gpu_stats']:
            gpu = stats['gpu_stats']
            print(f"\nGPU: {gpu['name']}")
            print(f"GPU Usage: {gpu['usage']:.1f}%")
            print(f"GPU Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%)")
            print(f"GPU Temperature: {gpu['temperature']}°C")
        else:
            print("\nGPU: Not available or not detected")
        
        # Training metrics
        if len(self.total_rewards) > 0:
            print(f"\nTraining Episodes: {len(self.total_rewards)}")
            print(f"Average Reward: {np.mean(self.total_rewards):.2f}")
            print(f"Latest Reward: {self.total_rewards[-1]:.2f}")
    
    def save_stats_to_file(self, filename=None):
        """Save statistics to JSON file"""
        if filename is None:
            filename = f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': time.time() - self.training_start_time if self.training_start_time else 0,
            'current_stats': self.get_current_stats(),
            'history': {
                'timestamps': list(self.timestamps),
                'gpu_usage': list(self.gpu_usage_history),
                'gpu_memory': list(self.gpu_memory_history),
                'gpu_temperature': list(self.gpu_temp_history),
                'cpu_usage': list(self.cpu_usage_history),
                'ram_usage': list(self.ram_usage_history)
            },
            'training_rewards': self.total_rewards
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Stats saved to: {filename}")
        return filename
    
    def plot_realtime_dashboard(self):
        """Create real-time monitoring dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Isaac Lab Training Monitor', fontsize=16)
        
        def animate(frame):
            if len(self.timestamps) > 0:
                # Convert timestamps to relative time
                current_time = time.time()
                times = [(current_time - t) / 60 for t in self.timestamps]  # minutes ago
                times.reverse()  # Show most recent on right
                
                # Clear previous plots
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                
                # GPU Usage
                if len(self.gpu_usage_history) > 0:
                    gpu_usage = list(self.gpu_usage_history)
                    gpu_usage.reverse()
                    ax1.plot(times, gpu_usage, 'b-', linewidth=2)
                    ax1.set_title('GPU Usage %')
                    ax1.set_ylabel('Usage %')
                    ax1.set_ylim(0, 100)
                    ax1.grid(True)
                
                # GPU Memory
                if len(self.gpu_memory_history) > 0:
                    gpu_memory = list(self.gpu_memory_history)
                    gpu_memory.reverse()
                    ax2.plot(times, gpu_memory, 'r-', linewidth=2)
                    ax2.set_title('GPU Memory %')
                    ax2.set_ylabel('Memory %')
                    ax2.set_ylim(0, 100)
                    ax2.grid(True)
                
                # CPU Usage
                if len(self.cpu_usage_history) > 0:
                    cpu_usage = list(self.cpu_usage_history)
                    cpu_usage.reverse()
                    ax3.plot(times, cpu_usage, 'g-', linewidth=2)
                    ax3.set_title('CPU Usage %')
                    ax3.set_xlabel('Minutes Ago')
                    ax3.set_ylabel('Usage %')
                    ax3.set_ylim(0, 100)
                    ax3.grid(True)
                
                # RAM Usage
                if len(self.ram_usage_history) > 0:
                    ram_usage = list(self.ram_usage_history)
                    ram_usage.reverse()
                    ax4.plot(times, ram_usage, 'm-', linewidth=2)
                    ax4.set_title('RAM Usage %')
                    ax4.set_xlabel('Minutes Ago')
                    ax4.set_ylabel('Usage %')
                    ax4.set_ylim(0, 100)
                    ax4.grid(True)
        
        ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
        plt.tight_layout()
        return fig, ani

def main():
    parser = argparse.ArgumentParser(description='Isaac Lab Training Analysis Tool')
    parser.add_argument('--mode', choices=['monitor', 'dashboard', 'stats'], default='monitor',
                      help='Mode: monitor (console), dashboard (GUI), stats (current stats)')
    parser.add_argument('--interval', type=float, default=1.0,
                      help='Update interval in seconds')
    parser.add_argument('--duration', type=int, default=None,
                      help='Monitoring duration in seconds')
    parser.add_argument('--save', action='store_true',
                      help='Save stats to file')
    
    args = parser.parse_args()
    
    analyzer = TrainingAnalyzer(update_interval=args.interval)
    
    try:
        if args.mode == 'stats':
            # Just print current stats
            analyzer.print_current_stats()
            
        elif args.mode == 'monitor':
            # Console monitoring
            analyzer.start_monitoring()
            
            start_time = time.time()
            try:
                while True:
                    analyzer.print_current_stats()
                    
                    if args.duration and (time.time() - start_time) >= args.duration:
                        break
                    
                    time.sleep(5)  # Update console every 5 seconds
                    
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
            
            analyzer.stop_monitoring()
            
            if args.save:
                analyzer.save_stats_to_file()
                
        elif args.mode == 'dashboard':
            # GUI dashboard
            analyzer.start_monitoring()
            
            try:
                fig, ani = analyzer.plot_realtime_dashboard()
                plt.show()
            except KeyboardInterrupt:
                print("\nDashboard stopped by user")
            
            analyzer.stop_monitoring()
            
            if args.save:
                analyzer.save_stats_to_file()
    
    except Exception as e:
        print(f"Error: {e}")
        analyzer.stop_monitoring()

if __name__ == "__main__":
    main()