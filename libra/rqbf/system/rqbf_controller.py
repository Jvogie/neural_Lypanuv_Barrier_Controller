## Rattle Quaternion Barrier Function Controller (Prototype)
## rqbf_controller.py

import random

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController

## Taken mostly from Charles work.
## Not used in final implementation or any iteration of the project

class HybridNeuralCLBFController(NeuralCLBFController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle_avoidance_weight = 0.5  ## Adjust this weight to balance CLBF and reactive avoidance

    def u(self, x: torch.Tensor) -> torch.Tensor:
        ## Get the control input from the neural CLBF
        u_clbf = super().u(x)
        
        ## Get the nominal control input (includes obstacle avoidance)
        u_nominal = self.dynamics_model.u_nominal(x)
        
        ## Combine the two control inputs
        u_combined = (1 - self.obstacle_avoidance_weight) * u_clbf + self.obstacle_avoidance_weight * u_nominal
        
        ## Clip the combined control input to respect control limits
        upper_u_lim, lower_u_lim = self.dynamics_model.control_limits
        u_combined = torch.clamp(u_combined, min=lower_u_lim, max=upper_u_lim)
        
        return u_combined
    
    def hybrid_loss(self, x: torch.Tensor, u_clbf: torch.Tensor, u_nominal: torch.Tensor) -> torch.Tensor:
        ## Compute a loss that encourages the hybrid control to be close to both CLBF and nominal control
        u_hybrid = self.u(x)
        loss_clbf = F.mse_loss(u_hybrid, u_clbf)
        loss_nominal = F.mse_loss(u_hybrid, u_nominal)
        return loss_clbf + loss_nominal

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        batch_dict = super().training_step(batch, batch_idx)
        
        x, _, _, _ = batch
        u_clbf = super().u(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        
        hybrid_loss = self.hybrid_loss(x, u_clbf, u_nominal)
        
        if optimizer_idx == 0:  ## CLBF optimizer
            batch_dict['loss'] += hybrid_loss
        elif optimizer_idx == 1:  ## Hybrid optimizer
            batch_dict['loss'] = hybrid_loss
        
        batch_dict['hybrid_loss'] = hybrid_loss
        
        return batch_dict

    def validation_step(self, batch, batch_idx):
        batch_dict = super().validation_step(batch, batch_idx)
        
        x, _, _, _ = batch
        u_clbf = super().u(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        
        hybrid_loss = self.hybrid_loss(x, u_clbf, u_nominal)
        batch_dict['val_loss'] += hybrid_loss
        batch_dict['hybrid_loss'] = hybrid_loss
        
        return batch_dict

    def configure_optimizers(self):
        clbf_optimizer = super().configure_optimizers()[0]
        
        ## Add any additional parameters specific to the hybrid controller
        hybrid_params = [p for p in self.parameters() if p.requires_grad]
        hybrid_optimizer = torch.optim.Adam(hybrid_params, lr=self.primal_learning_rate)
        
        return [clbf_optimizer, hybrid_optimizer]

    @pl.core.decorators.auto_move_data # type: ignore
    def simulator_fn(self, x_init: torch.Tensor, num_steps: int):
        ## Use the hybrid controller for simulation
        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=self.get_random_scenario(),
        )

    def get_random_scenario(self):
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return random_scenario