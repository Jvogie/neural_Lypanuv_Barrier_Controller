## Wrapper class for the 6DOF vehicle model in MuJoCo
## Was designed due to a horrific lack of performance of the original dynamics model in Mujoco.

import mujoco
import numpy as np
import torch
from libra.rqbf.system.rqbf import SixDOFVehicle
from scipy.spatial.transform import Rotation as R

class MuJocoSixDOFVehicle(SixDOFVehicle):
    def __init__(self, model_path, nominal_params, dt=0.01, controller_dt=None, scenarios=None):
        super().__init__(nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios)
        self.model = mujoco.MjModel.from_xml_path(model_path) ## type: ignore
        self.data = mujoco.MjData(self.model) ## type: ignore

    def reset(self, initial_state=None):
        mujoco.mj_resetData(self.model, self.data) ## type: ignore
        if initial_state is not None:
            self.data.qpos[:3] = initial_state[:3]  ## type: ignore
            ## Convert quaternion to euler angles
            r = R.from_quat(initial_state[3:7])
            euler = r.as_euler('xyz')
            self.data.qpos[3:6] = euler  ## type: ignore
            self.data.qvel[:3] = initial_state[7:10]  ## type: ignore
            self.data.qvel[3:6] = initial_state[10:13]  ## type: ignore
        mujoco.mj_forward(self.model, self.data) ## type: ignore
        return self.get_state()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data) ## type: ignore
        return self.get_state()

    def get_state(self):
        state = np.zeros(19)
        state[0:3] = self.data.qpos[:3]  # position
        
        ## Convert euler angles to quaternion
        euler = self.data.qpos[3:6]
        r = R.from_euler('xyz', euler)
        quat = r.as_quat()
        state[3:7] = quat  ## quaternion (w, x, y, z)
        
        state[7:10] = self.data.qvel[:3]  ## linear velocity
        state[10:13] = self.data.qvel[3:6]  ## angular velocity
        state[13:16] = 0.0  # accelerometer bias (assumed zero)
        state[16:19] = 0.0  ## gyroscope bias (assumed zero)
        return torch.tensor(state, dtype=torch.float32)
