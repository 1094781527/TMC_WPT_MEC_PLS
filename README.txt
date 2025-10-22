MEC-WPT-PLS Algorithm Comparison Platform
A mobile edge computing simulation platform integrating nine deep reinforcement learning algorithms.
Quick Start
Installation dependencies:
pip install torch numpy scipy matplotlib pandas tensorflow
Run the program:
python all_comparison.py
Including algorithms:
-OURs (our algorithm)
-PG (Policy Gradient)
- AC（Actor-Critic）
-DDPG (Deep Deterministic Policy Gradient)
-DROO (Deep Reinforcement Learning Optimization)
-H2AC (Mixed Actor Critic)
-A3C (Asynchronous Advantage Actor Critic)
-PPO (Near End Policy Optimization)
-SAC (Soft Actor Critic)
Basic usage:
Run with default parameters: python all_comparison.cy
Custom training steps: python all_comparison-py -- time steps 10000
Specify data file: python all_comparison.cy -- file-path your_data.mat
Output result:
-Console Performance Comparison Report
-Algorithm reward curve graph
Parameter configuration:
Modify the Config class in the code to adjust system parameters:
-N=15 (number of devices)
-Hyperparameters such as learning rate and batch size
Please refer to the code comments for detailed documentation