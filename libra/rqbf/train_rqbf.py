## Rattle Quaternion Barrier Function Training Script
## train_rqbf.py

from neural_clbf.monkeystrap import monkey_patch_distutils, monkey_patch_tqdm

monkey_patch_distutils()
monkey_patch_tqdm()

from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.training.utils import current_git_hash

## An attempt to continue of Charles controller with obstacle avoidance built into u, was never finished by him. I tried to do it myself but never got around to testing it properly as I kept running out of memory
## from libra.rbqf.rqbf_controller import HybridNeuralCLBFController
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
from libra.rqbf.system.rqbf import SixDOFVehicle
from libra.rqbf.eval_rqbf import RandomizeEnvironmentCallback

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05
simulation_dt = 0.01

def main(args):
    nominal_params = {
        "mass": 9.58,
        "inertia_matrix": torch.eye(3),
        "gravity": torch.tensor([0.0, 0.0, 0.0]),
    }

    ## Note, see systems/utils.py to see how things are solved when multiple scenarios are provided
    ## Also, rbqf.py has it's linearized controller override
    ## I don't think we should invest in multiple scenarios *yet* as it performs fine.
    ## We'd need to rework a lot as solving with multiple scenarios is a lot more robust and things become infeasible quickly.
    ## We'd probably have to rework the rqbf system to accomodate this.
    scenarios = [
        nominal_params,
    ## {"mass": 1.1, "inertia_matrix": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "gravity": torch.tensor([0.0, 0.0, 0.0])},
    ]

    ## It's going to be trained with a specific goal later, but you can modify this after training as seen in the experiment file
    goal_position = torch.tensor([0.0, 0.0, 0.0])

    initial_obstacles = [
        (torch.tensor([0.0, 2.5, -5.0]), 1.5),
        (torch.tensor([-2.5, -2.5, -5.0]), 1.5),
        (torch.tensor([0, 5, -5.0]), 1.5),
        (torch.tensor([0, -6.5, -7.0]), 1.5),
    ]

    dynamics_model = SixDOFVehicle(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
        obstacles=initial_obstacles,
    )
    dynamics_model.set_goal(goal_position)

    initial_conditions = [
        (-10.0, 10.0),   ## r_x
        (-10.0, 10.0),   ## r_y
        (-10.0, 10.0),   ## r_z
        (1.0, 1.0),      ## q_w
        (0.0, 0.0),      ## q_x
        (0.0, 0.0),      ## q_y
        (0.0, 0.0),      ## q_z
        (-5.0, 5.0),     ## v_x
        (-5.0, 5.0),     ## v_y
        (-5.0, 5.0),     ## v_z
        (-1.0, 1.0),     ## omega_x
        (-1.0, 1.0),     ## omega_y
        (-1.0, 1.0),     ## omega_z
        (-0.1, 0.1),     ## b_a_x
        (-0.1, 0.1),     ## b_a_y
        (-0.1, 0.1),     ## b_a_z
        (-0.1, 0.1),     ## b_g_x
        (-0.1, 0.1),     ## b_g_y
        (-0.1, 0.1),     ## b_g_z
    ]
    
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=100,
        fixed_samples=20000,
        max_points=200000,
        val_split=0.2,
        batch_size=128,
    )
    
    experiment_suite = ExperimentSuite([])

    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module, ## type: ignore
        experiment_suite=experiment_suite,
        clbf_hidden_layers=5,
        clbf_hidden_size=512,
        clf_lambda=1.0,
        safe_level=1.0,
        num_init_epochs=10,
        epochs_per_episode=100,
        barrier=True,
        disable_gurobi=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/six_dof_vehicle",
        name=f"commit_{current_git_hash()}",
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/six_dof_vehicle",
        filename="{epoch}-{val_loss:.2f}",
        save_last=True,
        save_top_k=50,
        monitor="val_loss",
        mode="min",
    )
    
    position_range = (20.0, 20.0, 20.0)
    num_obstacles = 4
    num_start_points = 4

    randomize_environment_callback = RandomizeEnvironmentCallback(
        dynamics_model, position_range, num_obstacles, num_start_points
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=1000,
        callbacks=[checkpoint_callback, randomize_environment_callback],
      ## trying to get cuda working, comment this out if you want to test gpu
      ##  gpus=1 if torch.cuda.is_available() else 0,
    )

    torch.autograd.set_detect_anomaly(True) ## type: ignore
    trainer.fit(clbf_controller)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("CUDA is available: ", torch.cuda.is_available())

    main(args)
