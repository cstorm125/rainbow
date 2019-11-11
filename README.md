# Rainbow

Implementation of [Rainbow paper](https://arxiv.org/abs/1710.02298) sans distributional RL as part of [Reinforcement Learning course at Chula 2019](https://github.com/ekapolc/RL_course_2019)

This repository contains the solution notebooks of all Rainbow improvements except distributional reinforcement learning. 

The `solutions` folder contains
* `environments.py` - The `SingleStockEnvironment` for single-asset trading (buy, hold, sell)
* `agents.py` - Agents for function-approximation Q-learning and deep Q-learning
* `networks.py` - Networks used by the agents
* `memories.py` - Memories used by the agents
* `utils.py` - Some utility functions

The notebooks are
* `qlearning_fa.ipynb` - Function-approximation Q-learning
* `dqn_vanilla.ipynb` - Vanilla deep Q-learning
* `dqn_double.ipynb` - Double deep Q-learning
* `dqn_prioritized.ipynb` - DQN with prioritized memory
* `dqn_nstep.ipynb` - DQN with N-step memory
* `dqn_dueling.ipynb` - DQN with dueling networks
* `dqn_noisy.ipynb` - DQN with noisy linear layers for exploration
* `dqn_rainbow.ipynb` - Rainbow implementation without distributional RL
* `dqn_trading.ipynb` - Use Rainbow to trade bitcoins