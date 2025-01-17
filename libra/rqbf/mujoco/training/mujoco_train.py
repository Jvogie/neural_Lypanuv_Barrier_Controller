## Finetuning the 6DOF vehicle model in MuJoCo

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.training.utils import current_git_hash
from libra.rqbf.mujoco.training.mujoco_wrapper import MuJocoSixDOFVehicle
from argparse import ArgumentParser
import yaml

class CustomCLBFDataset(Dataset):
    def __init__(self, states, inputs, next_states):
        self.states = states
        self.inputs = inputs
        self.next_states = next_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.inputs[idx], self.next_states[idx]

def generate_training_data(env, num_samples):
    states = []
    inputs = []
    next_states = []

    for _ in range(num_samples):
        state = env.reset()
        action = torch.rand(env.n_controls) * 40 - 20  # Random action between -20 and 20
        next_state = env.step(action)

        states.append(state)
        inputs.append(action)
        next_states.append(next_state)

    return torch.stack(states), torch.stack(inputs), torch.stack(next_states)

def custom_load_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    ## I believe callbacks broke for the finetuning process when Brendan was working on it, so this is a bandaid
    ## Could be useful later to add back in, to randomize scenarios between epochs for finetuning
    if 'callbacks' in checkpoint:
        del checkpoint['callbacks']
    
    model = NeuralCLBFController.load_from_checkpoint(checkpoint_path, strict=False)
    
    ## Manually load the state dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return model

class SafeYAMLEncoder(yaml.SafeDumper):
    def represent_data(self, data):
        if isinstance(data, dict):
            return self.represent_dict({str(k): v for k, v in data.items()})
        return super().represent_data(data)

def train_with_mujoco(model_path, checkpoint_path, batch_size=32):
    nominal_params = {
        "mass": torch.tensor(9.58),
        "inertia_matrix": torch.eye(3),
        "gravity": torch.tensor([0.0, 0.0, 0.0]),
    }

    env = MuJocoSixDOFVehicle(model_path, nominal_params, dt=0.05)

    ## Generate training data
    states, inputs, next_states = generate_training_data(env, 200000) 

    ## Create dataset and dataloader
    dataset = CustomCLBFDataset(states, inputs, next_states)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    controller = custom_load_from_checkpoint(checkpoint_path)

    ## Create a custom logger that doesn't save hparams
    class CustomLogger(pl.loggers.TensorBoardLogger): ## type: ignore
        def log_hyperparams(self, params):
            pass

    logger = CustomLogger(
        "logs/mujoco_six_dof_vehicle",
        name=f"commit_{current_git_hash()}",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/mujoco_six_dof_vehicle",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=1000,
        callbacks=[checkpoint_callback],
    )

    torch.autograd.set_detect_anomaly(True) ## type: ignore
    trainer.fit(controller, train_loader)


def main(args):
    model_path = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\neural_clbf\libra\rqbf\mujoco\astrobee.xml"
    checkpoint_path = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\six_dof_vehicle\epoch=0-val_loss=23.26.ckpt"
    train_with_mujoco(model_path, checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)