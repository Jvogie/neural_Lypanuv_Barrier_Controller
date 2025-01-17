# Neural Lyapunov Barrier Controller

This repository contains the results of a Neural Lyapunov Barrier Controller, which combines neural networks with control theory for safe and stable control of dynamical systems. The project specifically focuses on satellite control applications in space environments.

## Project Overview

This project implements and compares different control strategies for satellite attitude and position control in space:
- Neural Control Lyapunov Barrier Function (Neural-CLBF)
- Model Predictive Control (MPC)
- Linear Quadratic Regulator (LQR)

The controllers are designed to handle satellite maneuvering tasks and tested across various scenarios with increasing complexity to evaluate their performance and robustness in space applications.

## Application Context

This work focuses on developing and comparing control strategies for satellite systems in space, where:
- Controllers must operate in complex orbital dynamics
- Precise attitude and position control is critical
- Safe and efficient maneuvering is essential for mission success
- Control must be robust to uncertainties in the space environment

## Simulation Environment

All simulations are conducted using the Mujoco physics engine under zero gravity conditions to accurately represent the space environment. This setup allows for:
- Precise physics-based simulation of satellite systems
- Controlled testing environment without gravitational effects
- Accurate comparison between different control strategies in space-like conditions

## Repository Structure

- `Large-Scale-Training/`: Contains training results and simulation videos for different control approaches
  - Neural-CLBF controller runs
  - MPC controller runs
  - LQR controller runs
  
- `Increasing_complexity_sims/`: Contains simulation results for scenarios with increasing complexity
  - Progressive test cases from simple to complex scenarios
  - Comparative analysis between different control strategies

- `Plotly-Plots-of-Controllers/`: Visualization and analysis plots of controller performance

## Results

The project includes various visualization outputs:
- `IncreasingComplexity_Trends.png`: Shows performance trends across increasing complexity
- `Aggregated_Results_newplot_IncreasedComplexityTest_V2.0.png`: Aggregated performance analysis
- Multiple simulation videos demonstrating controller behavior in different scenarios

## Visualization Examples

Example visualizations are included:
- `plotly_example.png`: Sample visualization of controller performance

## Implementation Details

The project implements multiple control strategies and compares their performance:
1. Neural-CLBF: A neural network-based approach combining Control Lyapunov Functions with Barrier Functions
2. MPC: Traditional Model Predictive Control implementation
3. LQR: Classical Linear Quadratic Regulator approach

Each controller is tested across various scenarios with different complexity levels to evaluate their effectiveness and robustness.

## Libra Components

The project includes significant additions in the `libra` folder, which contains several key components:

### RBQF (Rattle Quaternion Barrier Function)
- A specialized controller designed for the Rattle system
- Uses quaternion-based representation for attitude control
- Includes training, experimentation, and evaluation scripts
- Features Mujoco physics integration for realistic simulation
- Known limitations:
  - Z-axis offset issues
  - Quaternion controllability challenges in the control affine system
  - Mujoco implementation currently shows steering discrepancies compared to direct evaluation

### Additional Components
- `RPO_Tests/`: Rendezous Proximity Operations test implementations
- `euler_attempt/`: Alternative approach using Euler angles
- `Other_controller/`: Additional control implementations

## Neural CLBF Implementation (found in `neural_clbf` folder)

The Neural Control Lyapunov Barrier Function (Neural CLBF) implementation provides:
- Integration of neural networks with control theory
- Safe and stable control of dynamical systems
- Training scripts for various scenarios (e.g., inverted pendulum)
- Evaluation tools and visualization capabilities
- Tensorboard integration for training monitoring

## Results and Analysis

The repository contains extensive simulation results and comparisons between different control strategies. The results are documented through:
- Video recordings of simulation runs
- Performance plots and visualizations
- Comparative analysis across different complexity levels
