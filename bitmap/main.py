import asyncio
import os
import sys

# Add the current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_env import RobotExplorationEnv

def get_map_path():
    """Get the map image path - update this for your system"""
    return r"C:\Users\HP\Desktop\Projects\navigation\9-daniel-cremers-random-motion-collect\environments\images\4.png"

def select_strategy():
    """Strategy selection menu"""
    print("\n=== Robot Exploration Strategy Selection ===")
    print("1. Random Walk")
    print("2. Lévy Walk")
    print("3. Manual Control")
    print("4. Custom Lévy Walk")
    
    while True:
        choice = input("\nSelect strategy (1-4): ").strip()
        if choice == "1":
            from strategies.random_walk import RandomWalkStrategy
            return RandomWalkStrategy()
        elif choice == "2":
            from strategies.levy_walk import LevyWalkStrategy
            return LevyWalkStrategy()
        elif choice == "3":
            from strategies.manual_control import ManualControlStrategy
            return ManualControlStrategy()
        elif choice == "4":
            from strategies.levy_walk import LevyWalkStrategy
            alpha = float(input("Enter alpha parameter (default 1.6): ") or "1.6")
            min_step = float(input("Enter min step (default 1.0): ") or "1.0")
            max_step = float(input("Enter max step (default 200.0): ") or "200.0")
            return LevyWalkStrategy(alpha=alpha, min_step=min_step, max_step=max_step)
        else:
            print("Invalid choice. Please select 1-4.")

def configure_environment():
    """Environment configuration"""
    print("\n=== Environment Configuration ===")
    max_steps = input("Enter max steps (default 1000000): ").strip()
    max_steps = int(max_steps) if max_steps else int(1000e3)
    
    render = input("Enable rendering? (y/n, default y): ").strip().lower()
    render = render != 'n'
    
    return max_steps, render

async def main():
    try:
        # Strategy selection
        strategy = select_strategy()
        
        # Environment configuration
        max_steps, render = configure_environment()
        
        # Create environment with strategy metadata
        env = RobotExplorationEnv(
            map_image_path=get_map_path(),
            robot_radius=5,
            render=render,
            max_steps=max_steps,
            strategy_name=strategy.name,
            strategy_parameters=strategy.parameters
        )
        
        print(f"\nStarting {strategy.name} strategy...")
        print(f"Parameters: {strategy.parameters}")
        print(f"Max steps: {max_steps}")
        print(f"Output directory: {env.output_dir}")
        
        # Run the strategy
        steps, coverage = await strategy.run(env)
        
        print(f"\n=== Simulation Complete ===")
        print(f"Final Coverage: {coverage:.2f}%")
        print(f"Total Steps: {steps}")
        print(f"Results saved to: {env.output_dir}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    asyncio.run(main())