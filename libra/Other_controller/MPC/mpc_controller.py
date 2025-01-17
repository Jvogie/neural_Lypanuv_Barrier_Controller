from scipy.optimize import minimize
import numpy as np
from numba import njit

@njit
def vectorized_cost(u, A, B, Q, R, N, x, goal, obstacle_positions, obstacle_radii):
    x_pred = x.copy()
    cost = 0.0
    for i in range(N):
        u_step = u[i*B.shape[1]:(i+1)*B.shape[1]]
        x_pred = A @ x_pred + B @ u_step
        state_error = x_pred - goal
        cost += state_error.T @ Q @ state_error + u_step.T @ R @ u_step
        for j in range(len(obstacle_positions)):
            distance = np.linalg.norm(x_pred[:3] - obstacle_positions[j])
            if distance < 2.0:
                cost += 100.0 / (distance + 1e-2)
    return cost

class MPCController:
    def __init__(self, A, B, Q, R, N=10):
        """
        Initialize the MPC controller.
        :param A: System dynamics matrix
        :param B: Input matrix
        :param Q: State cost matrix
        :param R: Control cost matrix
        :param N: Prediction horizon
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.N = N
        self.u_prev = np.zeros(B.shape[1])

    def compute_mpc_control(self, x, goal, obstacles):
        """
        Compute the MPC control input.
        :param x: Current state
        :param goal: Goal state
        :param obstacles: List of obstacles
        :return: Control input
        """
        obstacle_positions = np.array([obs.position for obs in obstacles])
        obstacle_radii = np.array([obs.radius for obs in obstacles])

        # Initial guess
        u0 = np.tile(self.u_prev, self.N)

        # Constraints
        bounds = [(-1, 1) for _ in range(self.N * self.B.shape[1])]

        # Optimize
        res = minimize(
            vectorized_cost,
            u0,
            args=(self.A, self.B, self.Q, self.R, self.N, x, goal, obstacle_positions, obstacle_radii),
            bounds=bounds,
            method='SLSQP'
        )

        if res.success:
            optimal_u = res.x[:self.B.shape[1]]
            self.u_prev = optimal_u
            return optimal_u
        else:
            print("MPC optimization failed, using previous control.")
            return self.u_prev
