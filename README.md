# Visual Navigation Urgent Circumstances via Counterfactual Reasoning in CARLA Simulator (VCCR)
This project is submitted on Robot Artificial Intelligence Class.

## Installation
1. Install the Carla Simulator in [CARLA official installation homepage](https://carla.readthedocs.io/en/latest/build_linux/).
2. Clone `VCCR` repository.
```
git clone --recursive https://github.com/brunoleej/VCCR.git
```
3. Create a Anaconda environment and install `VCCR` packages.
```
cd VCCR
conda env create -f environment/gpu-env.yml
conda activate VCCR
pip install -e .
```

## Usuage
Configure files can be found in `examples/config/`.
```
VCCR run_local examples.development --config=examples.config.
```
Currently only running locally is supported.
## New environments
To run on a different environment, you can modify the provided template. 

## Logging


## Hyperparameters


## Comparing to VCCR


## Reference
```
@inproceedings{janner2019mbpo,
  author = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
  title = {When to Trust Your Model: Model-Based Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

## Acknowledgements
The underlying Deep Deterministic Policy Gradient (DDPG), Soft-Actor-Critic (SAC), Twin-Delayed Deep Deterministic Policy Gradient (TD3) implementation in Experiment section comes from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) codebase.
