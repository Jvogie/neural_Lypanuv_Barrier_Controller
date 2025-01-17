# Controller base class to standardize interface
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class BaseController(ABC):
    @abstractmethod
    def get_control(self, state: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Get control input given current state, goal, and obstacles"""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Perform any necessary controller setup"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get controller name for logging and comparison"""
        pass