import numpy as np
import matplotlib.pyplot as plt
from robot_env import DownsampledRobotEnv, downsample_and_discretize

def test_downsampling():
    """Test the downsampling and discretization function."""
    print("Testing downsampling and discretization...")
    
    # Create test data
    np.random.seed(42)
    test_distances = np.random.uniform(0, 20, 100)
    
    # Test the function
    discretized = downsample_and_discretize(
        test_distances, 
        num_outputs=3, 
        num_levels=3, 
        max_range=20
    )
    
    print(f"Original 100 distances: {test_distances[:10]}... (showing first 10)")
    print(f"Downsampled to 3 values: {discretized}")
    print(f"Interpretation: 0=close, 1=medium, 2=far")
    
    # Show distribution
    print("\nDiscretization distribution:")
    for i, val in enumerate(discretized):
        print(f"  ray_{i}: level {val}")

def test_environment():
    """Test the environment with visualization."""
    print("\nTesting environment...")
    
    # Create environment with rendering enabled
    env = DownsampledRobotEnv("environments/images/12.png", render_mode=True, max_steps=100)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation (3 discretized rays): {obs}")
    print(f"Goal position: ({env.goal_x}, {env.goal_y})")
    print(f"Robot position: ({env.x}, {env.y})")
    
    # Take a few random steps
    for i in range(20):
        action = np.random.randint(0, 5)  # 0-4
        obs, reward, done, _ = env.step(action)
        
        # Render
        if not env.render():
            break
            
        print(f"Step {i+1}: Action={action}, Obs={obs}, Reward={reward}, Done={done}")
        
        if done:
            print("Episode finished!")
            break

def analyze_action_space():
    """Analyze the action space."""
    print("\nAction space analysis:")
    print("Action 0: Up/Forward")
    print("Action 1: Down/Backward")
    print("Action 2: Left (turn)")
    print("Action 3: Right (turn)")
    print("Action 4: Nothing (stay in place)")
    
    # Calculate possible observation space size
    # 3 rays, each with 3 levels: 3^3 = 27 possible observations
    print(f"\nObservation space size: 3^3 = {3**3} possible states")
    print("Each observation is a tuple of 3 integers (0, 1, or 2)")

if __name__ == "__main__":
    print("="*60)
    print("DOWN SAMPLED ENVIRONMENT TEST")
    print("="*60)
    
    test_downsampling()
    print("\n" + "-"*60)
    
    analyze_action_space()
    print("\n" + "-"*60)
    
    # Uncomment to test with visualization
    # test_environment()
    
    print("\nTest complete!")