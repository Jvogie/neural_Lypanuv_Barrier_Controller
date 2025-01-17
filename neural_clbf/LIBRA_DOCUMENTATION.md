# clone project or (download zip and unzip)
```bash
git clone https://github.com/dawsonc/neural_clbf
```

# download dependencies

## python

Make sure you have python 3.8.10

## visual studio c++ build tools

go to https://visualstudio.microsoft.com/visual-cpp-build-tools/

during installation, make sure to select the "Desktop development with C++" workload

# install project

## Note on linux

You'll likely run into errors regarding cvxpy, you are good to ignore this as long as it installs properly (which is should, tested on ubuntu)

If you tried previously, delete the old venv and re-install.

## Install dependencies

venv method

```bash
sudo apt-get install python3.8-venv
```

```bash
cd neural_clbf # go to root
python3 -m venv venv # create virtual environment
source venv/bin/activate # activate virtual environment
pip install -e .
pip install -r requirements.txt 
```

conda method

```bash
cd neural_clbf # go to root
conda create --name neural_clbf python=3.8.10 # create conda environment
conda activate neural_clbf # activate conda environment
pip install -e .
pip install -r requirements.txt
```

install neural_clbf (again)

Sometimes you need to install it again after installing the dependencies. Not sure why.

```bash
cd neural_clbf # assuming in project-libra, go to neural_clbf
pip install -e .
```

## To run

### Distutils monkey patch

If you run into any error regarding version checking for tensorboard, include the following in your training script:

```bash
from neural_clbf.monkeystrap import monkey_patch_distutils

monkey_patch_distutils()
```

### Train a model

in training/

pick a model, for example inverted_pendulum

```bash
python neural_clbf/training/train_inverted_pendulum.py
```

Next you can evaluate in evaluation/

!TIP: You'll probably have to change the path to the model in the eval script to match where you trained it, as it's currently hardcoded.

```bash
python neural_clbf/evaluation/eval_inverted_pendulum.py
```

### View training results

For example, if you trained the inverted pendulum model, you can view the results by running

```bash
tensorboard --logdir=logs/inverted_pendulum
```

## Specific Libra Documentation

### The Libra Folder

Inside the neural_clbf/libra folder, you will find all major additions to neural_clbf.

Other slight changes have been made to the neural_clbf codebase to accomodate the Libra additions. Nothing major.

#### RBQF

The Rattle Quaternion Barrier Function is a collection of scripts that is what will hopefully become the controller for rattle.

it have the relevant training, experimentation, and evaluation scripts inside

#### Known Issues with RBQF

Z axis is almost always incorrect by a fixed amount

Quaternions are 'uncontrollable' in the RBQF system, this cannot be fixed right now. We either need to switch to euler and see if that fixes it, or figure a way to make them controllable in the control affine system.

##### Mujoco

The Mujoco physics application is inside the rbqf/mujoco folder.

Currently a work in progress, but a way to have the controller actually interact with a physics system.

Right now it does not seem to be applying it the same way as the rbqf evaluation script, doesn't seem to be direct. I assume I am applying the controller incorrectly somewhere.

As a result it steers and misses quite a bit, compared to the evaluation script which is direct.



