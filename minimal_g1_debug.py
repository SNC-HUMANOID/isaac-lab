#!/usr/bin/env python3

"""
Minimal G1 Debug Version - หาปัญหาที่ทำให้ล่ม
"""

import torch
import numpy as np
import argparse
import time

# Isaac Lab imports
import isaaclab
from isaaclab.app import AppLauncher

# Minimal arguments
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--cpu", action="store_true", default=False)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launch
from isaaclab.envs import ManagerBasedRLEnv

try:
    import omni.isaac.lab_tasks
    from omni.isaac.lab_tasks.utils import load_cfg_from_registry
    TASKS_AVAILABLE = True
except ImportError:
    TASKS_AVAILABLE = False
    print("❌ Lab tasks not available")

class MinimalG1Debug:
    """Minimal G1 for debugging crashes"""
    
    def __init__(self):
        self.env = None
        self.step_count = 0
        self.crash_step = None
        
    def create_safe_environment(self):
        """Create environment with safe settings"""
        try:
            print("🔧 Creating MINIMAL G1 environment...")
            
            if not TASKS_AVAILABLE:
                print("❌ Cannot create G1 environment - tasks not available")
                return False
            
            # Load config
            env_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-G1-v0", "env_cfg_entry_point")
            
            # SAFE SETTINGS
            env_cfg.scene.num_envs = 1  # Only 1 environment
            env_cfg.sim.device = "cpu" if args_cli.cpu else "cuda:0"
            
            # Reduce physics complexity
            env_cfg.sim.dt = 0.01  # Larger timestep
            env_cfg.sim.substeps = 1  # Fewer substeps
            
            # Disable problematic features
            if hasattr(env_cfg, 'events'):
                # Minimize random events
                for event_cfg in env_cfg.events.__dict__.values():
                    if hasattr(event_cfg, 'mode') and hasattr(event_cfg, 'params'):
                        if hasattr(event_cfg.params, 'interval_range_s'):
                            event_cfg.params.interval_range_s = (1000.0, 1000.0)  # Very rare
            
            # Safe viewer settings  
            if not args_cli.headless:
                env_cfg.viewer.eye = [5.0, 5.0, 3.0]
                env_cfg.viewer.lookat = [0.0, 0.0, 0.0]
            
            # Create environment
            self.env = ManagerBasedRLEnv(cfg=env_cfg)
            
            print("✅ Minimal environment created")
            print(f"   Action space: {self.env.action_space.shape}")
            print(f"   Observation space: {self.env.observation_space}")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_safe_actions(self):
        """Generate very conservative actions"""
        action_dim = self.env.action_space.shape[-1]
        
        # VERY SMALL ACTIONS to prevent crashes
        actions = torch.zeros(1, action_dim)
        
        # Add tiny sinusoidal motion only to safe joints
        phase = self.step_count * 0.01  # Very slow
        
        if action_dim >= 37:
            # Only move safe joints with tiny amplitudes
            actions[0, 0] = 0.05 * np.sin(phase)      # Left hip pitch - tiny motion
            actions[0, 1] = 0.05 * np.sin(phase + np.pi)  # Right hip pitch - tiny motion
            
            # All other joints stay at 0 (safe)
        
        return actions
    
    def run_debug_test(self, max_steps=500):
        """Run debug test to find crash point"""
        if not self.env:
            print("❌ No environment")
            return
            
        print(f"\n🔍 DEBUG TEST - Running {max_steps} steps")
        print("   🎯 Goal: Find where/why it crashes")
        print("   ⚡ Using minimal safe actions")
        
        try:
            # Reset environment
            print("📍 Step 0: Resetting environment...")
            observations, _ = self.env.reset()
            print(f"✅ Reset successful. Obs shape: {observations['policy'].shape}")
            
            for step in range(max_steps):
                self.step_count = step
                
                # Progress reporting
                if step % 50 == 0:
                    print(f"📍 Step {step}: Running...")
                
                # Generate safe actions
                actions = self.generate_safe_actions()
                
                # DEBUG: Print action info occasionally
                if step % 100 == 0:
                    print(f"   🎲 Actions shape: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
                
                # CRITICAL: Step environment with error handling
                try:
                    observations, rewards, terminated, truncated, info = self.env.step(actions)
                    
                    # Check for problems
                    if torch.isnan(rewards).any():
                        print(f"⚠️  NaN reward at step {step}")
                    
                    if torch.isnan(observations['policy']).any():
                        print(f"⚠️  NaN observation at step {step}")
                        
                    # Auto reset if terminated
                    if terminated.any() or truncated.any():
                        print(f"🔄 Environment reset at step {step}")
                        observations, _ = self.env.reset()
                        
                except Exception as e:
                    print(f"💥 CRASH at step {step}: {e}")
                    self.crash_step = step
                    raise e
                
                # Brief pause to prevent overwhelming
                if step % 100 == 0:
                    time.sleep(0.1)
            
            print(f"✅ DEBUG TEST COMPLETED - No crashes in {max_steps} steps!")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Test stopped by user at step {self.step_count}")
            
        except Exception as e:
            print(f"\n💥 CRASHED at step {self.step_count}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Crash analysis
            print(f"\n🔍 CRASH ANALYSIS:")
            print(f"   Crash step: {self.crash_step or self.step_count}")
            print(f"   Last action shape: {actions.shape if 'actions' in locals() else 'Unknown'}")
            print(f"   Environment state: {'Created' if self.env else 'Not created'}")
    
    def run_memory_test(self):
        """Test for memory leaks"""
        print("\n🧠 MEMORY TEST")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Run many short episodes
        for episode in range(10):
            try:
                observations, _ = self.env.reset()
                
                for step in range(50):
                    actions = self.generate_safe_actions()
                    observations, rewards, terminated, truncated, info = self.env.step(actions)
                    
                    if terminated.any() or truncated.any():
                        break
                
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Episode {episode}: {current_memory:.1f} MB (+{current_memory - initial_memory:.1f})")
                
            except Exception as e:
                print(f"Memory test crashed at episode {episode}: {e}")
                break
    
    def close(self):
        """Clean up"""
        if self.env:
            self.env.close()

def main():
    """Main debug function"""
    print("🔍 G1 DEBUG MODE")
    print("=" * 40)
    
    debugger = MinimalG1Debug()
    
    try:
        # Test 1: Environment creation
        print("\n1️⃣ Testing Environment Creation...")
        if not debugger.create_safe_environment():
            print("❌ Environment creation failed - cannot continue")
            return
        
        # Test 2: Basic functionality
        print("\n2️⃣ Testing Basic Functionality...")
        debugger.run_debug_test(max_steps=200)
        
        # Test 3: Memory usage
        print("\n3️⃣ Testing Memory Usage...")
        debugger.run_memory_test()
        
    except Exception as e:
        print(f"🚨 MAJOR ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        debugger.close()
        
    print("\n🔚 Debug session complete")
    simulation_app.close()

if __name__ == "__main__":
    main()