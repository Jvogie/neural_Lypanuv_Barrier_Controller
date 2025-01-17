import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
import heapq

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

class LQRController:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self.solve_dare(A, B, Q, R)
        self.path = None
        self.waypoint_index = 0

    def solve_dare(self, A, B, Q, R):
        P = linalg.solve_discrete_are(A, B, Q, R)
        return np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    def plan_path(self, start, goal, obstacles, resolution=1.0):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def get_neighbors(node):
            neighbors = []
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                next_node = (node[0] + dx * resolution, node[1] + dy * resolution, node[2] + dz * resolution)
                if all(-10 <= coord <= 10 for coord in next_node):  # Assuming a 20x20x20 space
                    neighbors.append(next_node)
            return neighbors

        start_node = tuple(start[:3])
        goal_node = tuple(goal[:3])

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: heuristic(start_node, goal_node)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if np.linalg.norm(np.array(current) - np.array(goal_node)) < resolution:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return np.array(path[::-1])

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + resolution

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("No path found")
        return np.array([start[:3], goal[:3]])

    def get_control(self, x, obstacles, goal):
        # x is a 12-dimensional state vector: [pos, orientation, vel, ang_vel]
        pos = x[:3]
        orientation = x[3:6]
        vel = x[6:9]
        ang_vel = x[9:12]

        # Desired state: goal position, zero orientation, zero velocity, zero angular velocity
        desired_state = np.concatenate([goal[:3], np.zeros(3), np.zeros(6)])

        # Compute error
        error = x - desired_state  # [pos - goal_pos, ori - 0, vel - 0, ang_vel - 0]

        # Compute LQR control input
        u_lqr = -self.K @ error  # Negative feedback

        # Compute obstacle avoidance force
        obstacle_force = self.compute_obstacle_avoidance(pos, vel, obstacles)

        # Combine LQR control with obstacle avoidance
        alpha = 0.8  # Slightly reduced weight for LQR control
        u = alpha * u_lqr[:3] + (1 - alpha) * obstacle_force  # Translations
        u = np.concatenate([u, u_lqr[3:]])  # Keep orientation control from LQR

        # Limit the maximum control input to match actuator ranges
        max_control = 1.0  # actuator ctrlrange is [-1, 1]
        u = np.clip(u, -max_control, max_control)

        return u

    def compute_obstacle_avoidance(self, pos, vel, obstacles):
        total_force = np.zeros(3)
        influence_distance = 2.0  # Reduced influence distance
        max_repulsion_strength = 5.0  # Reduced repulsion strength

        for obstacle in obstacles:
            distance = obstacle.distance(pos)
            if distance < influence_distance:
                repulsion_strength = max_repulsion_strength * (1 - distance / influence_distance) ** 2
                force_direction = pos - obstacle.position
                norm = np.linalg.norm(force_direction) + 1e-6
                force_direction /= norm
                repulsive_force = repulsion_strength * force_direction
                
                # Add velocity-dependent component
                vel_alignment = np.dot(vel, -force_direction)
                if vel_alignment > 0:
                    repulsive_force += 2 * vel_alignment * force_direction  # Reduced velocity-dependent component

                total_force += repulsive_force

        return total_force

def create_astrobee_lqr():
    A = np.eye(12)
    dt = 0.01
    # Position to velocity
    A[0:3, 6:9] = np.eye(3) * dt
    # Orientation to angular velocity
    A[3:6, 9:12] = np.eye(3) * dt

    B = np.zeros((12, 6))
    B[6:9, 0:3] = np.eye(3) * dt  # Translational control
    B[9:12, 3:6] = np.eye(3) * dt  # Rotational control

    # Adjusted Q and R matrices
    Q = np.eye(12)
    Q[0:3, 0:3] *= 10    # Position error weight reduced
    Q[3:6, 3:6] *= 1     # Orientation error weight reduced
    Q[6:9, 6:9] *= 1     # Velocity error weight
    Q[9:12, 9:12] *= 1   # Angular velocity error weight

    R = np.eye(6) * 1.0    # Control effort cost increased

    return LQRController(A, B, Q, R)
