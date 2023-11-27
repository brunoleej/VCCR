# Visual Navigation under Urgent Circumstances via Counterfactual Reasoning in CARLA Simulator (VCCR)
This project is submitted on Robot Artificial Intelligence Class.

## Baseline model
- Model-based Offline Policy Optimization
- Causal Inference

## Running Environment
We are currently working on below setup.
- Ubuntu: `Ubuntu 20.04`
- Python: `3.8.18`
- Carla Simulator: `latest`

## Installation
1. Install the Carla Simulator in [CARLA official installation](https://carla.readthedocs.io/en/latest/build_linux/).
2. Clone `VCCR` repository.
```
git clone --recursive https://github.com/brunoleej/VCCR.git
```
3. Create a Anaconda environment and run `requirement.txt` file.
```
cd VCCR
conda env create -f venv_setup.yml
conda activate vccr
pip install -e viskit
pip install -e .
```

## Usage
To run our baseline algorithm, execute files can be found in `vccr/MOPO/train_mopo.py`
```
VCCR train_mopo.py --config=cfg.env.yaml
```


Configure files can be found in `examples/config/`.
```
VCCR run_local examples.development --config=examples.config.
```
Currently only running locally is supported.
### New environments
To run on a different environment, you can modify the provided template. 

### Logging
For now Wandb logging is automatically running. If you don't mind you can comment processing the wandb logging line

### Hyperparameters


## Comparing to VCCR


## References
```
@inproceedings{janner2019mbpo,
  author = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
  title = {When to Trust Your Model: Model-Based Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}

@article{causal-rl-survey,
  title={Causal Reinforcement Learning: A Survey},
  author={Deng, Zhihong and Jiang, Jing and Long, Guodong and Zhang, Chengqi},
  journal={arXiv preprint arXiv:2307.01452},
  year={2023}
}
```

## Acknowledgements
The underlying Deep Deterministic Policy Gradient (DDPG), Soft-Actor-Critic (SAC), Twin-Delayed Deep Deterministic Policy Gradient (TD3) implementation in Experiment section comes from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) codebase.
