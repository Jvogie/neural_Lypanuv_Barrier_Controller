# Documentation for the RQBF (Rattle Quaternion Barrier Function) Controller

## Notable Commits

https://github.com/Orbital-Services-Corporation/Project-Libra/commit/df18e3a246f01f4eda152f013d213dae40fe04df

First commit of brendan adding obstacles

Addition of obstacle buffers
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/1872a8d640ff3aa5eb8713645c326fcc15d57040

Smother trajectories:
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/bcfc9cdd2ca1e96e04ccd8f5a79e9fa16c0fe804

Mujoco with obstacles:
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/e802361f8d19727ba8674dc16756398480009b1a

added mujoco fine tuning
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/cb82c83d4f8ddd36edfed93ed20db74377494357

Improved fine tuning
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/e4e8d3d3dca721c79c838ef29e0c684ea71d5ef3

Refactored u-nominal
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/bf80db6fa46f79a2cda854590793df2e2c298326

Everything after that point was mostly just me trying to get it to work, only to realize it was inherently flawed

### Commits where it is the "best"

These would be the commits on October 14th, 2024 - October 15th, 2024

https://github.com/Orbital-Services-Corporation/Project-Libra/commit/e4e8d3d3dca721c79c838ef29e0c684ea71d5ef3
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/bf80db6fa46f79a2cda854590793df2e2c298326
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/5ffc90962d6a56661f887deaaf75599ba7f4bb4b
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/01a56ad2a72ab88ca8ced8bcd782d022d4b0cea6
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/348846751078b5c9f6c79888d4fd7211535518e7
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/5b65f2a0e79ce803384accba5f5c8a8570f787f9
https://github.com/Orbital-Services-Corporation/Project-Libra/commit/324c551bf04fd67eb87620a47a22b543fa85508b

All at various points of working-ness

## Notable files:

### Strictly related to RQBF:

- `neural_clbf/libra/rbqf/rqbf.py`
- `neural_clbf/libra/rbqf/rqbf_controller.py`
- `neural_clbf/libra/rbqf/rqbf_experiment.py`
- `neural_clbf/libra/rbqf/train_rqbf.py`
- `neural_clbf/libra/rbqf/eval_rqbf.py`

### Fine-tuning:

- `neural_clbf/libra/rbqf/mujoco_train.py`
- `neural_clbf/libra/rbqf/mujoco_wrapper.py`

### Simulation:

- `neural_clbf/libra/rbqf/mujoco/main_3d.py`
- `neural_clbf/libra/rbqf/mujoco/main_2d.py`
- `neural_clbf/libra/rbqf/mujoco/astrobee.xml`

### For reference:

- `neural_clbf/neural_clbf/systems/control_affine_system.py`
- `neural_clbf/neural_clbf/controllers/neural_clbf_controller.py`
- `neural_clbf/neural_clbf/controllers/neural_cbf_controller.py`
- `neural_clbf/neural_clbf/controllers/bf_controller.py`
- `neural_clbf/neural_clbf/controllers/controller.py`

## Overview

### The Current RQBF System

### The RQBF (Rattle Quaternion Barrier Function) System

The RQBF system is an attempt to create a control barrier function for a 6-DOF vehicle using quaternions for orientation representation. It's implemented primarily in the following files:

- `neural_clbf/libra/rbqf/rqbf.py`
- `neural_clbf/libra/rbqf/train_rqbf.py`

#### Key Components

1. **State Space**: The system uses a 19-dimensional state space:
   - Position (3D)
   - Quaternion orientation (4D)
   - Linear velocity (3D)
   - Angular velocity (3D)
   - Accelerometer bias (3D)
   - Gyroscope bias (3D)

2. **Control Inputs**: 6-dimensional control input:
   - Force (3D)
   - Torque (3D)

3. **Dynamics**: The system implements control-affine dynamics, following the structure of the `ControlAffineSystem` base class.

4. **Obstacle Avoidance**: The system includes obstacle avoidance mechanisms, but these have proven problematic in practice.

5. **Training**: The system is trained using the `NeuralCLBFController`, with attempts to incorporate obstacle avoidance directly into the nominal control. Attempts were made to use the `HybridNeuralCLBFController`, but this was not successful as we do not have the compute/memory to run it.

#### Issues and Limitations

1. **Quaternion Representation**: The use of quaternions for orientation, while theoretically sound, has led to practical issues. The system often reverts to Euler angles in simulation when necessary, which defeats the purpose of using quaternions and may introduce singularities.

2. **Obstacle Avoidance**: Despite multiple attempts with different approaches (including distance-based, velocity-based, and angle-based masks), the system has consistently failed to properly avoid obstacles.

3. **U-Nominal Controller**: Various implementations of the nominal controller have been attempted, ranging from basic PD controllers to sophisticated nonlinear PD controllers with obstacle avoidance. None have proven consistently effective.

4. **Linearization Issues**: The system overrides the `compute_linearized_controller` method due to persistent infeasibility issues with the computed LQR controller in early versions.

5. **Memory Issues**: Attempts to implement a more sophisticated controller (`HybridNeuralCLBFController`) have been hampered by memory limitations during testing.

6. **Training Stability**: The system shows signs of instability during training, with no clear indication that it's learning effectively. Overfitting/Underfitting have been observed in various points of development.

7. **Lack of Diverse Scenarios**: The training process does not include a wide variety of scenarios, limiting the system's ability to generalize and handle different situations effectively. We had planned to add more but it led to infeasibility. This occurs for reasons detailed in `train_rqbf.py`

#### Hyperparameters

1. **System Parameters**:
   - Mass: 9.58
   - Inertia Matrix: Identity matrix (3x3)
   - Gravity: [0.0, 0.0, 0.0]

2. **Simulation Parameters**:
   - Simulation timestep (dt): 0.01
   - Controller period: 0.05

3. **Neural Network Architecture**:
   - Hidden layers: 5
   - Hidden size: 256
   - Input size: 19 (state dimension)
   - Output size: 1 (CLBF value)

4. **Training Parameters**:
   - Batch size: 128
   - Learning rate: 1e-3 (primal_learning_rate in NeuralCLBFController)
   - Number of initial epochs: 5
   - Epochs per episode: 100
   - Maximum epochs: 1000
  
It rarely gets trained above 10 epochs. Most tests are done at 1-5 epochs.

5. **CLBF Parameters**:
   - CLF lambda: 1.0
   - Safe level: 1.0
   - CLF relaxation penalty: 50.0

6. **Data Generation**:
   - Trajectories per episode: 100
   - Trajectory length: 100
   - Fixed samples: 20000
   - Maximum points: 200000
   - Validation split: 0.2

7. **Environment Randomization**:
   - Position range: (20.0, 20.0, 20.0)
   - Number of obstacles: 4
   - Number of start points: 4

8. **Optimization**:
   - Optimizer: SGD
   - Weight decay: 1e-6

These hyperparameters can be found and adjusted in the `train_rqbf.py` file and the `NeuralCLBFController` class.

### Simulation/Evaluation

#### RQBF_Experiment

The `SixDOFVehicleRolloutExperiment` class in `rqbf_experiment.py` is designed to simulate and visualize trajectories of the 6-DOF vehicle under the RQBF controller. It include:

1. **Simulation**: Runs multiple simulations with different starting points, tracking the vehicle's state, control inputs, and barrier function values.

2. **Obstacle Handling**: Detects collisions with obstacles and terminates simulations upon collision.

3. **Goal Reaching**: Checks if the vehicle reaches the specified goal position.

4. **Visualization**: Generates 3D plots of the trajectories, including:
   - Start points
   - End points (with different markers for collision, goal reached, or active)
   - Obstacles as transparent spheres
   - Goal position (if specified)

5. **Data Logging**: Records detailed information about each simulation step, including time, position, barrier function value, control inputs, and simulation status.

##### Results

After removing the obstacle bounce-off behavior, the vehicle will consistently fail to avoid obstacles and reach the goal. If there is no obstacle in the way it will reach the goal. It does struggle with direct head-on collisions, and does not appear to be able to learn to avoid obstacles in general.

#### MuJoCo 3D Simulation (main_3d.py)

The `main_3d.py` file implements a 3D simulation environment using MuJoCo for the RQBF controller. It include:

1. **MuJoCo Integration**: Uses MuJoCo for physics simulation and visualization.

2. **GLFW Window**: Creates an interactive GLFW window for real-time visualization.

3. **Camera Control**: Implements mouse and keyboard controls for camera movement.

4. **Dynamic Obstacle Generation**: Generates random obstacles along the path from start to goal.

5. **XML Configuration**: Dynamically modifies the Astrobee XML file to include generated obstacles.

6. **Multiple Agent Simulation**: Simulates multiple agents with different starting positions.

7. **Performance Metrics**: Computes and logs performance metrics for each agent's trajectory.

8. **3D Trajectory Visualization**: Generates an interactive 3D plot of all agent trajectories using Plotly.

##### Results

The simulation results show that:

1. The controller struggles with obstacle avoidance. If there is an obstacle in the direct path, it will hit it directly.
2. Agents can reach the goal when there are no obstacles in the direct path. But will frequently make near misses. It often takes 1-2 attempts to hit the goal. This could be solved with more epochs perhaps. Or a better u-nominal controller.

### Conclusion

It doesn't work.

Many issues have arisen, the main being that it never has been able to properly avoid obstacles.

Even early on we were actually having trouble to get it to properly go to obstacle in Mujoco, it would swerve and completely miss before timing out.

It performed better in the `rqbf_experiment.py` file, but even there it was not able to avoid obstacles properly, What appears as obstacle avoidance is actually bumping into the obstacle and then veering away.

It's not clear that it's even learning anything.

### Direct changes to the original code

Some changes have been ignored as they were trivial or not related to the issues.

#### Changes that did not have any effect

1. In `neural_clbf/neural_clbf/systems/f16.py` and `neural_clbf/neural_clbf/experiments/__init__.py` we silenced some import warnings.

#### Changes that likely have no effect on our issues

1. In `setup.py` and `neural_clbf/neural_clbf/utils.py` we fixed a crash that occurs if not ran in a GitHub Repository.
2. In `requirements.txt` we initally changed some requirements to get it working for linux, this seems to have been fine as we could still train the originals authors code. We later changed these even more to be more specific with a pip freeze.
3. In `neural_clbf/neural_clbf` we added `monkeystrap.py` to monkey patch some fixes, related to tqdm and distutils. This was necessary as some of the code was not compatible with the new versions of these libraries. It is not likely that this is the cause of any issues currently.
4. In `neural_clbf/neural_clbf/experiments/experiment_suite.py` we changed `run_all_and_log_plots` to better handle `rqbf_experiment.py`.
5. In `neural_clbf/neural_clbf/controllers/neural_clbf_controller.py` we moved the `HybridNeuralCLBFController` to `rqbf_controller.py` and modified it to be a direct implementation of the controller.

#### Changes that may have had an effect

1. We changed all occurrences of gurobi being used to default to false and use cvxpy instead. Seeing as how the original others used gurobi, this could possible have an effect.

In other words, we do not believe that any of these changes are the cause of the issues.

### Possible Causes of Failure

#### The rqbf system is inherently unstable/flawed

##### It's Just Not Controllable

The biases may not be relevant or controllable and may not be cut, same with quaternions. They could be removed and changed to Euler angles and later converted back.

##### Quaternions

This is quite likely actually, we chose to use quaternions to describe orientation, which could be a fundamental flaw. This was ignored early on by just not dealing with orientation, and later ignored even more by converting to euler where necessary, despite the fact that we retained quaternions in the state space.

##### Safe/Unsafe Masks

We have tried many different masks, none of which have worked.

1. Simple distance threshold with relying on u-nominal to avoid obstacles
2. More advanced mask based on distance to obstacles and velocity
3. Mask based on distance to obstacles and angle between velocity and the vector to the obstacle.

#### U-Nominal

It's quite possible that we don't have the right u-nominal. We've tried from basic PD controllers to way more advanced ones that implements a sophisticated nonlinear PD controller with obstacle avoidance.

The current one computes the nominal control input for the vehicle based on its current state and goal position. 

The controller uses position and velocity errors to generate desired accelerations, with adaptive gains that increase precision near the goal. It incorporates obstacle avoidance using a repulsive force that adapts based on proximity to obstacles. 

The function also includes orientation control to align the vehicle with the desired acceleration vector.

To prevent oscillations and improve stability, it implements an integral term for position error and a progress-based reduction of obstacle avoidance effects. 
 
The controller applies speed limits and control input clamping to ensure the outputs stay within the system's physical limitations.

So basically we've attempted everything in between a very basic PD controller to a very advanced nonlinear PD controller with obstacle avoidance. There could be something fundamentally wrong with the approach, and we should try something else and use maybe LQR, MPC, or even point-mass LQR.

#### The Overridden Linear Controller

Early on we ended up deciding to override the `compute_linearized_controller` method in the `control_affine_system` class due to constant unfeasibility of the computed LQR controller.

The `rqbf.py` implementation is more specialized, focusing on a subset of the state space (controllable indices) for LQR computation and adjusting weights for bias states. It uses only continuous-time dynamics matrices and expands the LQR result to the full state space. In contrast, the `control_affine_system.py` version is more generalized, using the full state space for LQR computation with uniform cost matrices. It incorporates both continuous-time and discrete-time dynamics matrices in its calculations.

#### The Neural CLBF Controller

So the actual implementation of the Neural CLBF controller doesn't *seem* to specifically encourage obstacle avoidance. It trusts this to be inherently be learned.

This could be a major part of why it doesn't work. There was another controller called `HybridNeuralCLBFController` that was attempted, but was barebones and not finished. We turned this into `rqbf_controller.py` where we had to implement it all from scratch. Issue is, we run out of memory whenever we test it.

#### Or it's simply too complicated to be used with Neural CLBF

It could be that our system is simply too complicated to be used with Neural CLBF. There is a lot of state space that is not controllable, and thus not able to be learned.

#### Lack of Diverse Training Scenarios

The training process does not include a wide variety of scenarios, which limits the system's ability to generalize and handle different situations effectively. This lack of diversity in training data could be a significant factor in the system's poor performance, especially when faced with new or complex obstacle configurations.

### Possible Solutions

Observe that the solution may be none of these, all of these, or a combination of these.

1. **Simplify the State Space**: 
   - Remove biases from the state space.
   - Consider switching from quaternions to Euler angles for orientation representation, at least during training.
   - Focus on a reduced state space that includes only position, velocity, and simplified orientation.

2. **Revise the U-Nominal Controller**:
   - Implement a simpler, more robust nominal controller. Although this may ruin the whole idea to begin with.
   - Consider using a point-mass LQR as a starting point and gradually increase complexity.
   - Separate obstacle avoidance from the nominal controller and handle it in the CLBF. This however is tricky as we've tried to do this in `rqbf_controller.py` and it ran into memory issues.

3. **Revisit the Linearized Controller**:
   - Re-evaluate the decision to override the `compute_linearized_controller` method.
   - Investigate why the original LQR controller was consistently infeasible and address those issues. Likely due to bullet point 1.
   - Consider using alternative linearization techniques or adaptive linearization.

4. **Enhance Obstacle Avoidance in Neural CLBF**:
   - Modify the Neural CLBF implementation to explicitly encourage obstacle avoidance. Ties in with bullet point 2.
   - Incorporate obstacle information directly into the CLBF loss function. `Done in rqbf_controller.py` but again runs into memory issues.
   - Experiment with different representations of obstacles in the state space.

5. **Optimize for Computational Efficiency**:
   - Refactor the code to reduce memory usage, allowing for testing of more complex controllers like the `HybridNeuralCLBFController`.
   - Implement batch processing or online learning techniques to handle large state spaces more efficiently.

6. **Gradual Complexity Increase**:
   - Start with a simpler version of the system (e.g., 2D or 3D without orientation).
   - Incrementally add complexity, ensuring each step works before moving to the next.
   - Validate the CLBF approach on simpler, known-controllable systems before applying it to the full 6-DOF vehicle.

7. **Alternative Learning Approaches**:
   - Explore other learning methodologies that might be more suitable for complex systems, such as reinforcement learning or model-based learning.
   - Consider hybrid approaches that combine traditional control theory with learning-based methods.

8. **Improved Safe/Unsafe Masks**:
   - Develop more sophisticated safe/unsafe masks that consider not just current state but predicted future states.
   - Incorporate dynamic obstacle prediction into the mask generation.
   - Use optimization techniques to generate masks that balance safety and reachability.

9. **Diversify Training Scenarios**:
   - Implement a more comprehensive set of training scenarios that cover a wide range of obstacle configurations and initial conditions.
   - Include edge cases and challenging situations in the training data to improve the system's ability to generalize.
   - Consider using procedural generation techniques to create a large variety of training environments automatically.

### Notes

If we somehow pick this back up in length, investing in making it work on cuda and more complex training mechanisms would be a good idea.