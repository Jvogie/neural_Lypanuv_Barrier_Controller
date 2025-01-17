import mujoco
import mujoco.viewer
import numpy as np
import time
from satellite_env import SatelliteEnv

def main():
    # Create the environment
    env = SatelliteEnv()
    
    # Reset the environment
    obs, _ = env.reset()

    # Create a MuJoCo viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Run the simulation for 5000 steps
        for i in range(5000):
            # Sample a random action (replace this with your controller later)
            action = env.action_space.sample()
            
            # Step the environment
            obs, reward, done, _, _ = env.step(action)
            
            # Update the viewer
            viewer.sync()
            
            # Print some information every 100 steps
            if i % 100 == 0:
                agent_pos = obs[:3]
                target_pos = obs[6:9]
                distance = np.linalg.norm(agent_pos - target_pos)
                print(f"Step {i}: Agent pos: {agent_pos}, Target pos: {target_pos}, Distance: {distance:.2f}")
            
            # Check if the episode is done
            if done:
                print("Episode done, resetting environment")
                obs, _ = env.reset()

            # Add a small delay to slow down the visualization
            time.sleep(0.01)

        # Keep the viewer open until the user closes it
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
