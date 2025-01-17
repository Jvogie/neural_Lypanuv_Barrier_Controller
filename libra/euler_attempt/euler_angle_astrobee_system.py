from typing import List, Tuple, Optional

import torch
import numpy as np
from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
import math


class AstrobeeSystem(ControlAffineSystem):
    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )
        self.goal_state = torch.zeros(1, self.n_dims)
        # self.n_controls = 6

    # ... (other methods remain the same)

    @property
    def n_dims(self) -> int:
        return 18  # 3 pos, 3 vel, 3 euler angles, 3 ang_vel, 3 accel_bias, 3 gyro_bias

    @property
    def angle_dims(self) -> List[int]:
        return [6, 7, 8]  # Euler angle dimensions

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([1.5, 6.4, 1.7, 0.1, 0.1, 0.1, 
                                    math.pi, math.pi/2, math.pi,  # Euler angles
                                    1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        lower_limit = -upper_limit
        lower_limit[7] = -math.pi/2  # Pitch angle limit
        return (upper_limit, lower_limit)
    

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        force_limit = 1.0  # Newtons, slightly increased
        torque_limit = 0.1  # Newton-meters, slightly increased
        upper_limit = torch.tensor([force_limit, force_limit, force_limit, 
                                    torque_limit, torque_limit, torque_limit])
        lower_limit = -upper_limit
        return (upper_limit, lower_limit)

    @property
    def n_controls(self):
        # Return the number of control inputs
        return 6  # Adjust this number based on your specific Astrobee model

    def validate_params(self, params):
        # Update the method signature to accept 'params'
        # Implement the validation logic here
        return True
        # Add more validation as needed


    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        r, v, _, _, _, _ = self._unpack_state(x)
        position_safe = torch.all(torch.abs(r) < torch.tensor([1.5, 6.4, 1.7], device=x.device), dim=1)
        velocity_safe = torch.all(torch.abs(v) < 0.1, dim=1)
        return position_safe & velocity_safe

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return ~self.safe_mask(x)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        r, v, _, _, _, _ = self._unpack_state(x)
        at_goal_position = torch.all(torch.abs(r) < 0.1, dim=1)
        at_goal_velocity = torch.all(torch.abs(v) < 0.05, dim=1)
        return at_goal_position & at_goal_velocity


    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), device=x.device)
        
        r, v, euler, omega, accel_bias, gyro_bias = self._unpack_state(x)
        
        f[:, 0:3, 0] = v
        euler_dot = self._euler_rates(euler, omega)
        f[:, 6:9, 0] = euler_dot
        
        # Add acceleration bias to velocity derivative
        f[:, 3:6, 0] = accel_bias
        
        # Add gyro bias to angular velocity derivative
        f[:, 9:12, 0] = gyro_bias
        
        return f

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls), device=x.device)
        
        # Assuming a constant mass of 9.58 kg for Astrobee
        mass = 9.58
        
        # Force input to acceleration
        g[:, 3:6, 0:3] = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1) / mass
        
        # Assuming a constant inertia tensor for Astrobee
        inertia = torch.tensor([0.153, 0.143, 0.162], device=x.device)
        
        r, v, euler, omega, _, _ = self._unpack_state(x)
        for i in range(batch_size):
            euler_i = euler[i]
            R = self._euler_to_rotation_matrix(euler_i)
            I_body = R @ torch.diag(inertia) @ R.t()
            
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-8
            I_body_inv = torch.inverse(I_body + epsilon * torch.eye(3, device=x.device))
            
            g[i, 9:12, 3:6] = I_body_inv
        
        return g

    def _unpack_state(self, x: torch.Tensor):
        return x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12], x[:, 12:15], x[:, 15:18]

    def _euler_to_quaternion(self, euler: torch.Tensor) -> torch.Tensor:
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack((w, x, y, z), dim=1)

    def _quaternion_to_euler(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack((roll, pitch, yaw), dim=1)

    def _euler_to_rotation_matrix(self, euler: torch.Tensor) -> torch.Tensor:
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)]
        ], device=euler.device)
        
        Ry = torch.tensor([
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)]
        ], device=euler.device)
        
        Rz = torch.tensor([
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1]
        ], device=euler.device)
        
        return Rz @ Ry @ Rx

    def _euler_rates(self, euler: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        roll, pitch, _ = euler[:, 0], euler[:, 1], euler[:, 2]
        wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
        
        roll_dot = wx + torch.sin(roll) * torch.tan(pitch) * wy + torch.cos(roll) * torch.tan(pitch) * wz
        pitch_dot = torch.cos(roll) * wy - torch.sin(roll) * wz
        yaw_dot = (torch.sin(roll) / torch.cos(pitch)) * wy + (torch.cos(roll) / torch.cos(pitch)) * wz
        
        return torch.stack((roll_dot, pitch_dot, yaw_dot), dim=1)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack((w, x, y, z), dim=1)

    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-6
        q = q / (torch.norm(q) + epsilon)  # Normalize with epsilon
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        return torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], device=q.device, dtype=torch.float32)

    def compute_A_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        x0 = self.goal_point
        u0 = self.u_eq
        
        def dynamics(x):
            return self.closed_loop_dynamics(x, u0, scenario).squeeze()
        
        A = torch.autograd.functional.jacobian(dynamics, x0).squeeze().cpu().numpy()
        
        # Add a small perturbation to ensure non-zero entries
        epsilon = 1e-6
        A += np.random.randn(*A.shape) * epsilon
        
        print("A matrix:")
        print(A)
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            print("Warning: A matrix contains NaN or inf values")
            A = np.nan_to_num(A, nan=0.0, posinf=1e30, neginf=-1e30)
        return A

    def compute_B_matrix(self, scenario: Optional[Scenario]) -> np.ndarray:
        B = super().compute_B_matrix(scenario)
        print("B matrix:")
        print(B)
        if np.any(np.isnan(B)) or np.any(np.isinf(B)):
            print("Warning: B matrix contains NaN or inf values")
            B = np.nan_to_num(B, nan=0.0, posinf=1e30, neginf=-1e30)
        return B
