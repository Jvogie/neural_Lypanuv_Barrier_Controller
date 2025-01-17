## Rollout Experiment for Rattle Quaternion Barrier Function System (RQBF)
## rqbf_experiment.py

## Overridden Experiment class for RQBF, was meant for graphing the trajectories on a 3d space for initial testing

import os
import torch
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

from neural_clbf.experiments import Experiment
from neural_clbf.systems.utils import ScenarioList
from libra.rqbf.system.rqbf import SixDOFVehicle
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController

class SixDOFVehicleRolloutExperiment(Experiment):
    """An experiment for simulating and plotting 6-DoF vehicle trajectories."""

    def __init__(
        self,
        name: str,
        start_x: torch.Tensor,
        scenarios: Optional[ScenarioList] = None,
        n_sims_per_start: int = 1,
        t_sim: float = 5.0,
        goal_position: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the SixDOFVehicleRolloutExperiment.

        Args:
            name (str): Name of the experiment.
            start_x (torch.Tensor): Initial states for the simulations.
            scenarios (Optional[ScenarioList]): List of scenarios to simulate.
            n_sims_per_start (int): Number of simulations per starting state.
            t_sim (float): Total simulation time.
            goal_position (Optional[torch.Tensor]): Goal position for the vehicle.
        """
        super().__init__(name)
        self.start_x = start_x
        self.scenarios = scenarios if scenarios is not None else []
        self.n_sims_per_start = n_sims_per_start
        self.t_sim = t_sim
        self.goal_position = goal_position
        
    @torch.no_grad()
    def run(self, controller_under_test: NeuralCLBFController) -> pd.DataFrame:
        """
        Run the experiment simulations.

        Args:
            controller_under_test (NeuralCLBFController): The controller being tested.

        Returns:
            pd.DataFrame: DataFrame containing simulation results.
        """
        ## Set up the simulation parameters
        delta_t = controller_under_test.dynamics_model.dt
        num_timesteps = int(self.t_sim // delta_t)
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls

        ## Prepare initial states
        n_sims = self.n_sims_per_start * self.start_x.shape[0]
        x_sim_start = self.start_x.repeat(self.n_sims_per_start, 1)
        x_current = x_sim_start.clone()
        x_current = x_current.to(controller_under_test.device)

        ## If a goal position is provided, set it in the dynamics model
        if self.goal_position is not None:
            controller_under_test.dynamics_model.set_goal(self.goal_position) ## type: ignore

        ## Prepare data storage
        results = []

        ## Keep track of active simulations
        active_sims = torch.ones(n_sims, dtype=torch.bool, device=controller_under_test.device)

        ## Simulation loop
        for tstep in tqdm.trange(num_timesteps, desc="Simulating trajectories"):
            ## Get control inputs for active simulations
            u_current = controller_under_test.u(x_current[active_sims])

            ## Log data for active simulations
            for sim_index in range(n_sims):
                if active_sims[sim_index]:
                    log_data = {
                        "t": tstep * delta_t,
                        "sim": sim_index,
                        "$x$": x_current[sim_index, 0].item(),
                        "$y$": x_current[sim_index, 1].item(),
                        "$z$": x_current[sim_index, 2].item(),
                        "V": controller_under_test.V(x_current[sim_index : sim_index + 1]).item(),
                        "control_input": u_current[active_sims[:sim_index+1].sum()-1].cpu().numpy(),
                        "state": x_current[sim_index].cpu().numpy(),
                        "status": "active"
                    }
                    results.append(log_data)

            ## Update state for active simulations
            xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                x_current[active_sims], u_current
            )
            x_current[active_sims] = x_current[active_sims] + delta_t * xdot

            ## Check for collisions with obstacles
            for obstacle_pos, obstacle_radius in controller_under_test.dynamics_model.obstacles: ## type: ignore
                distances = torch.norm(x_current[:, :3] - obstacle_pos, dim=1)
                collisions = distances <= obstacle_radius
                for sim_index in range(n_sims):
                    if active_sims[sim_index] and collisions[sim_index]:
                        active_sims[sim_index] = False
                        results.append({
                            "t": (tstep + 1) * delta_t,
                            "sim": sim_index,
                            "$x$": x_current[sim_index, 0].item(),
                            "$y$": x_current[sim_index, 1].item(),
                            "$z$": x_current[sim_index, 2].item(),
                            "V": controller_under_test.V(x_current[sim_index : sim_index + 1]).item(),
                            "control_input": np.zeros(n_controls),
                            "state": x_current[sim_index].cpu().numpy(),
                            "status": "collision"
                        })

            ## Check for reaching the goal
            if self.goal_position is not None:
                goal_distances = torch.norm(x_current[:, :3] - self.goal_position, dim=1)
                goal_reached = goal_distances <= 0.1  ## Assuming 0.1 as the threshold for reaching the goal
                for sim_index in range(n_sims):
                    if active_sims[sim_index] and goal_reached[sim_index]:
                        active_sims[sim_index] = False
                        results.append({
                            "t": (tstep + 1) * delta_t,
                            "sim": sim_index,
                            "$x$": x_current[sim_index, 0].item(),
                            "$y$": x_current[sim_index, 1].item(),
                            "$z$": x_current[sim_index, 2].item(),
                            "V": controller_under_test.V(x_current[sim_index : sim_index + 1]).item(),
                            "control_input": np.zeros(n_controls),
                            "state": x_current[sim_index].cpu().numpy(),
                            "status": "goal_reached"
                        })

            ## Break the loop if all simulations are terminated
            if not active_sims.any():
                break

        ## Convert results to DataFrame
        traj_df = pd.DataFrame(results)

        # After running the experiment, plot the results
        fig = self.plot(controller_under_test, traj_df)

        return traj_df

    def plot(self, controller_under_test: NeuralCLBFController, traj_df: pd.DataFrame, display_plots: bool = True):
        """
        Plot the 6-DoF vehicle trajectories.

        Args:
            controller_under_test (NeuralCLBFController): The controller being tested.
            traj_df (pd.DataFrame): DataFrame containing simulation results.
            display_plots (bool): Whether to display the plots interactively.

        Returns:
            list of tuples: Each tuple contains the plot name and the figure handle.
        """
        ## Plot the trajectories in 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ## Plot each trajectory
        for sim_index in traj_df['sim'].unique():
            sim_data = traj_df[traj_df['sim'] == sim_index]
            color = plt.cm.jet(sim_index / len(traj_df['sim'].unique())) ## type: ignore
            ax.plot3D(
                sim_data["$x$"],
                sim_data["$y$"],
                sim_data["$z$"],
                color=color,
                label=f"Trajectory {sim_index+1}"
            )
            
            ## Plot end points with different markers based on status
            last_point = sim_data.iloc[-1]
            if last_point['status'] == 'collision':
                ax.scatter(last_point["$x$"], last_point["$y$"], last_point["$z$"], c='r', s=100, marker='x')
            elif last_point['status'] == 'goal_reached':
                ax.scatter(last_point["$x$"], last_point["$y$"], last_point["$z$"], c='g', s=100, marker='*')
            else:
                ax.scatter(last_point["$x$"], last_point["$y$"], last_point["$z$"], c='b', s=100, marker='o')

        ## Plot start points
        start_points = traj_df.groupby('sim').first()
        ax.scatter(start_points["$x$"], start_points["$y$"], start_points["$z$"], c='k', s=100, marker='^', label='Start')

        ## Add the goal position if available
        if self.goal_position is not None:
            ax.scatter(self.goal_position[0], self.goal_position[1], self.goal_position[2], c='g', s=200, marker='*', label='Goal')

        ## Plot obstacles
        if isinstance(controller_under_test.dynamics_model, SixDOFVehicle):
            for obstacle_pos, obstacle_radius in controller_under_test.dynamics_model.obstacles:
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obstacle_pos[0] + obstacle_radius * np.outer(np.cos(u), np.sin(v))
                y = obstacle_pos[1] + obstacle_radius * np.outer(np.sin(u), np.sin(v))
                z = obstacle_pos[2] + obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='r', alpha=0.3)

        ## Set plot limits (adjust as needed)
        ax.set_xlim([-15.0, 15.0])
        ax.set_ylim([-15.0, 15.0])
        ax.set_zlim([-15.0, 15.0])
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ax.set_zlabel("$z$ (m)")

        ax.legend()
        plt.title("6-DoF Vehicle Trajectories under Neural CLBF Controller with Obstacles")
        
        if display_plots:
            plt.show()
        
        if hasattr(self, 'output_dir') and self.output_dir: ## type: ignore
            plot_file_path = os.path.join(self.output_dir, "trajectory_plot.png") ## type: ignore
            plt.savefig(plot_file_path)
            print(f"Trajectory plot saved to {plot_file_path}")

        ## Return a list with a single tuple (plot_name, figure_handle)
        return [("6-DoF Vehicle Trajectories with Obstacles", fig)]
