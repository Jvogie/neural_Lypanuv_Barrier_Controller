# Ensure this file contains the Obstacle class as used in MPC_controller.py
import numpy as np

class Obstacle:
    def __init__(self, position, size, shape='box'):
        self.position = np.array(position)
        self.size = np.array(size)
        self.shape = shape

    def distance(self, point):
        if self.shape == 'box':
            diff = np.abs(point - self.position) - self.size
            return np.linalg.norm(np.maximum(diff, 0)) + min(np.max(diff), 0)
        elif self.shape == 'sphere':
            return np.linalg.norm(point - self.position) - self.size[0]
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")
