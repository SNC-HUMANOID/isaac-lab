#!/usr/bin/env python3

import subprocess
import sys
import os
import signal
import argparse
from pathlib import Path

class TrainingMonitorWrapper:
    def __init__(self):
        self.training_process = None
        self.monitor_process = None
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\nShutting down training and monitoring...")
        if self.training_process:
            self.training_process.terminate()
        if self.monitor_process:
            self.monitor_process.terminate()
        sys.exit(0)
    
    def run_training_with_monitor(self, training_command, monitor_args=None):
        """Run training command with monitoring"""
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Default monitor arguments
        if monitor_args is None:
            monitor_args = ['--mode', 'monitor', '--save']
        
        try:
            # Start monitoring process
            monitor_cmd = [sys.executable, 'training_analysis.py'] + monitor_args
            print(f"Starting monitor: {' '.join(monitor_cmd)}")
            self.monitor_process = subprocess.Popen(monitor_cmd)
            
            # Start training process
            print(f"Starting training: {' '.join(training_command)}")
            self.training_process = subprocess.Popen(training_command)
            
            # Wait for training to complete
            return_code = self.training_process.wait()
            
            # Stop monitoring
            if self.monitor_process:
                self.monitor_process.terminate()
                self.monitor_process.wait()
            
            print(f"Training completed with return code: {return_code}")
            return return_code
            
        except Exception as e:
            print(f"Error running training with monitor: {e}")
            if self.training_process:
                self.training_process.terminate()
            if self.monitor_process:
                self.monitor_process.terminate()
            return 1

def main():
    parser = argparse.ArgumentParser(description='Isaac Lab Training with Monitoring')
    parser.add_argument('--task', required=True, help='Training task name')
    parser.add_argument('--framework', default='rsl_rl', choices=['rsl_rl', 'rl_games', 'sb3', 'skrl'],
                      help='RL framework to use')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--monitor-mode', default='monitor', choices=['monitor', 'dashboard'],
                      help='Monitoring mode')
    parser.add_argument('--monitor-interval', type=float, default=1.0,
                      help='Monitor update interval in seconds')
    parser.add_argument('--save-stats', action='store_true', default=True,
                      help='Save monitoring stats to file')
    
    args = parser.parse_args()
    
    wrapper = TrainingMonitorWrapper()
    
    # Build training command
    training_script = f"scripts/reinforcement_learning/{args.framework}/train.py"
    training_cmd = [
        "./isaaclab.sh", "-p", training_script,
        "--task", args.task
    ]
    
    if args.headless:
        training_cmd.append("--headless")
    
    # Build monitor arguments
    monitor_args = [
        '--mode', args.monitor_mode,
        '--interval', str(args.monitor_interval)
    ]
    
    if args.save_stats:
        monitor_args.append('--save')
    
    # Check if training script exists
    if not os.path.exists(training_script):
        print(f"Error: Training script not found: {training_script}")
        return 1
    
    print("="*60)
    print("ISAAC LAB TRAINING WITH MONITORING")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Framework: {args.framework}")
    print(f"Monitor Mode: {args.monitor_mode}")
    print(f"Training Command: {' '.join(training_cmd)}")
    print("="*60)
    
    # Run training with monitoring
    return wrapper.run_training_with_monitor(training_cmd, monitor_args)

if __name__ == "__main__":
    sys.exit(main())