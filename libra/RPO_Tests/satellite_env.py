import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class SatelliteEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Load the MuJoCo model from the XML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "satellite_model.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Set safe distance
        self.safe_distance = 2.0

        # Set target satellite movement parameters
        self.target_velocity = np.zeros(3)
        self.max_target_acceleration = 0.05

    def step(self, action):
        # Apply action to agent satellite (increase force)
        self.data.ctrl[:] = action * 5.0  # Increased force multiplier

        # Update target satellite movement
        self.update_target_movement()

        # Simulate one step
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Calculate reward
        agent_pos = obs[:3]
        target_pos = obs[6:9]
        distance = np.linalg.norm(agent_pos - target_pos)
        reward = 1.0 / (1.0 + abs(distance - self.safe_distance))

        # Check if done
        done = distance < 0.5 * self.safe_distance or distance > 2 * self.safe_distance

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial positions
        target_pos = self.np_random.uniform(low=-5, high=5, size=3)
        agent_pos = target_pos + self.np_random.uniform(low=-5, high=5, size=3)

        # Set initial positions in simulation
        self.data.qpos[:3] = agent_pos
        self.data.qpos[3:6] = target_pos

        # Set initial velocities (non-zero)
        self.data.qvel[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.target_velocity = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.data.qvel[3:6] = self.target_velocity

        return self._get_obs(), {}

    def update_target_movement(self):
        # Add some random acceleration to the target satellite
        acceleration = self.np_random.uniform(low=-self.max_target_acceleration, high=self.max_target_acceleration, size=3)
        self.target_velocity += acceleration
        
        # Apply the velocity to the target satellite
        self.data.qvel[3:6] = self.target_velocity

    def render(self):
        # Implement rendering if needed
        pass

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:3],  # agent position
            self.data.qvel[:3],  # agent velocity
            self.data.qpos[3:6],  # target position
            self.data.qvel[3:6]  # target velocity
        ])
