import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree

class RRTNode:
    def __init__(self, position):
        self.position = position
        self.parent: Optional[RRTNode] = None

class RRT:
    def __init__(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]], 
                 bounds: Tuple[np.ndarray, np.ndarray], max_iterations: int = 5000, step_size: float = 0.3):
        self.start = RRTNode(start[:3])
        self.goal = RRTNode(goal[:3])
        self.obstacles = [(obs[:3], rad) for obs, rad in obstacles]
        self.bounds = (bounds[0][:3], bounds[1][:3])
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.nodes = [self.start]
        self.kdtree = None

    def random_state(self):
        if np.random.random() < 0.1:
            return self.goal.position
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def nearest_node(self, point):
        if self.kdtree is None:
            self.kdtree = cKDTree([node.position for node in self.nodes])
        _, index = self.kdtree.query(point)
        return self.nodes[index]

    def new_state(self, from_point, to_point):
        direction = to_point - from_point
        length = np.linalg.norm(direction)
        if length > self.step_size:
            direction = direction / length * self.step_size
        return from_point + direction

    def is_collision_free(self, from_point, to_point):
        direction = to_point - from_point
        length = np.linalg.norm(direction)
        unit_vector = direction / length
        from_to_dot = np.dot(unit_vector, from_point)
        
        for obstacle_pos, obstacle_radius in self.obstacles:
            t = np.dot(unit_vector, obstacle_pos) - from_to_dot
            closest_point = from_point + np.clip(t, 0, length) * unit_vector
            if np.linalg.norm(closest_point - obstacle_pos) <= obstacle_radius + 0.5:
                return False
        return True

    def direct_path_possible(self):
        return self.is_collision_free(self.start.position, self.goal.position)

    def plan(self):
        if self.direct_path_possible():
            return self.smooth_path([self.start.position, self.goal.position])

        for _ in range(self.max_iterations):
            random_point = self.random_state()
            nearest = self.nearest_node(random_point)
            new_point = self.new_state(nearest.position, random_point)
            
            if self.is_collision_free(nearest.position, new_point):
                new_node = RRTNode(new_point)
                new_node.parent = nearest
                self.nodes.append(new_node)
                self.kdtree = None
                
                if np.linalg.norm(new_point - self.goal.position) < self.step_size:
                    self.goal.parent = new_node
                    return self.smooth_path(self.get_path())
        
        return None

    def get_path(self):
        path = []
        node = self.goal
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(self.start.position)
        return list(reversed(path))

    def smooth_path(self, path):
        smoothed_path = [path[0]]
        i = 0
        path_len = len(path)
        while i < path_len - 1:
            for j in range(path_len - 1, i, -1):
                if self.is_collision_free(path[i], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    break
            i += 1
        return smoothed_path

def rrt_plan(start: torch.Tensor, goal: torch.Tensor, obstacles: List[Tuple[torch.Tensor, float]], 
             bounds: Tuple[torch.Tensor, torch.Tensor], max_iterations: int = 5000, step_size: float = 0.3):
    start_np = start.cpu().numpy()
    goal_np = goal.cpu().numpy()
    obstacles_np = [(obs.cpu().numpy(), rad) for obs, rad in obstacles]
    bounds_np = (bounds[0].cpu().numpy(), bounds[1].cpu().numpy())

    rrt = RRT(start_np, goal_np, obstacles_np, bounds_np, max_iterations, step_size)
    path = rrt.plan()

    if path is not None:
        return torch.tensor(path, dtype=torch.float32, device=start.device)
    else:
        return None
