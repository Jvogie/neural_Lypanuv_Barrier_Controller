from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class BaseController(ABC):
    @abstractmethod
    def setup(self):
        """Initialize any necessary components for the controller"""
        pass

    @abstractmethod
    def get_control(self, state: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Compute control input given current state and environment
        
        Args:
            state: Current state vector
            goal: Goal state vector
            obstacles: List of (position, radius) tuples for obstacles
            
        Returns:
            Control input vector
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the controller"""
        pass 