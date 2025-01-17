from neural_clbf.monkeystrap import monkey_patch_distutils

monkey_patch_distutils()

from argparse import ArgumentParser
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from libra.euler_attempt.euler_angle_astrobee_system import AstrobeeSystem
from neural_clbf.experiments import ExperimentSuite, CLFContourExperiment, RolloutStateSpaceExperiment
from neural_clbf.training.utils import current_git_hash
import math
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05
simulation_dt = 0.01

def main(args):
    ## Define the scenarios
    nominal_params = {
        "force": torch.tensor([0.0, 0.0, 0.0]),  # N
        "torque": torch.tensor([0.0, 0.0, 0.0])  # N*m
    }
    scenarios = [
        nominal_params,
        ## Add more scenarios here if needed for robustness
    ]

    ## Define the dynamics model
    dynamics_model = AstrobeeSystem(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    ## Initialize the DataModule
    # relaxed initial conditions
    initial_conditions = [
        (-1.5, 1.5),  ## x position
        (-6.4, 6.4),  ## y position
        (-1.7, 1.7),  ## z position
        (-0.2, 0.2),  ## x velocity (relaxed)
        (-0.2, 0.2),  ## y velocity (relaxed)
        (-0.2, 0.2),  ## z velocity (relaxed)
        (-math.pi, math.pi),   ## roll (relaxed)
        (-math.pi/2, math.pi/2),  ## pitch (relaxed)
        (-math.pi, math.pi),   ## yaw (relaxed)
        (-1.5, 1.5),  ## angular velocity (3 dimensions, relaxed)
        (-1.5, 1.5),
        (-1.5, 1.5),
        (-0.2, 0.2),  ## accelerometer bias (3 dimensions, relaxed)
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, 0.2),  ## gyroscope bias (3 dimensions, relaxed)
        (-0.2, 0.2),
        (-0.2, 0.2),
    ]


    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    ## Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.5, 1.5), (-6.4, 6.4)],
        n_grid=30,
        x_axis_index=0,  ## x position
        y_axis_index=1,  ## y position
        x_axis_label="x",
        y_axis_label="y",
        plot_unsafe_region=False,
    )
    
    start_x = torch.tensor([
        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0] + [0.0] * 12,
        [-0.2, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 12,
        [0.2, -1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 12,
        [-0.2, -1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 12,
    ])
    
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        0,  ## x position
        "x",
        1,  ## y position
        "y",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    ## Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=4,  # Increased from 2
        clbf_hidden_size=128,  # Increased from 64
        clf_lambda=0.15,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e3,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=True,
        disable_gurobi=True,
    )

    # clbf_controller = NeuralCLBFController(
    #     dynamics_model,
    #     scenarios,
    #     data_module,
    #     experiment_suite=experiment_suite,
    #     clbf_hidden_layers=3,  # Reduced from 4
    #     clbf_hidden_size=96,   # Reduced from 128
    #     clf_lambda=0.01,       # Reduced from 0.05
    #     safe_level=1.0,
    #     controller_period=controller_period,
    #     clf_relaxation_penalty=1e3,  # Reduced from 1e4
    #     num_init_epochs=10,    # Increased from 5
    #     epochs_per_episode=150,  # Increased from 100
    #     barrier=True,
    #     disable_gurobi=True
    # )

    ## Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/astrobee",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=1,
    )

    ## Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)