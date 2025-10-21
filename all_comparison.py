from __future__ import print_function
import random
import time
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import tensorflow as tf
import argparse
from torch.distributions import Categorical
import math
import torch
import numpy as np
import scipy.io as sio
import copy
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, deque

# 超参数配置
N = 15  # 节点数
M = 1  # AP数
T = 20  # 单个时隙的长度
STATE_DIM = N * M
ACTION_DIM = N
ACTION_BOUND = 1.0
ACTION_PER_NODE = M + 1
JOINT_ACTION_DIM = N * ACTION_PER_NODE
STATE_DIM_A3C = 120
STATE_DIM_PPO = 120
STATE_DIM_SAC = 120
# DDPG参数
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.00001
CRITIC_LR = 0.0001
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
EXPLORE_NOISE = 0.1

# DROO参数
DROO_MEMORY_SIZE = 1024
DROO_DELTA = 32

# H2AC参数
H2AC_BATCH_SIZE = 32
H2AC_GAMMA = 0.99
H2AC_LR_POLICY = 0.0001
H2AC_LR_CRITIC = 0.0001
H2AC_LR_ACTOR = 0.001

# A3C参数
A3C_GAMMA = 0.99
A3C_LR_ACTOR = 1e-4
A3C_LR_CRITIC = 1e-3
A3C_ENTROPY_COEF = 0.01
A3C_MAX_GRAD_NORM = 0.5

# PPO参数
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_LR_ACTOR = 1e-4
PPO_LR_CRITIC = 1e-3
PPO_EPOCHS = 8
PPO_MINI_BATCH = 64
PPO_UPDATE_TIMESTEP = 1024
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5

# SAC参数
SAC_ALPHA = 0.2
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_LR = 3e-4
SAC_HIDDEN_DIM = 256

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser("arguments")
    # Environment
    parser.add_argument("--file-path", type=str, default="data.mat", help="file path for reading config and saving result")
    parser.add_argument("--max-episode-len", type=int, default=1, help="maximum episode length")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="min epsilon greedy")
    parser.add_argument("--min_noise_rate", type=float, default=0.05, help="min noise rate")
    parser.add_argument("--lagrangian_multiplier", type=float, default=1, help="initial value of lagrangian multiplier")
    parser.add_argument("--lagrangian_max_bound", type=float, default=20, help="max bound of lagrangian multiplier")
    parser.add_argument("--cost_threshold", type=float, default=2, help="threshold of cost")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lr-lagrangian", type=float, default=1e-3, help="learning rate of lagrangian multiplier")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="roundabout_env_result", help="directory in which training state and model should be saved")
    args = parser.parse_args()
    return args

class Leader:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.l_multiplier = args.lagrangian_multiplier
        self.cost_threshold = args.cost_threshold
        self.train_step = 0

        # create the network
        self.actor_network = Actor_Leader(args, agent_id, 35)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor_Leader(args, agent_id, 40)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))

        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

        if os.path.exists(self.model_path + '/cost_params.pkl'):
            self.cost_network.load_state_dict(torch.load(self.model_path + '/cost_params.pkl'))
            print('Agent {} successfully loaded cost_network: {}'.format(self.agent_id,
                                                                         self.model_path + '/cost_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.cost_target_network.parameters(), self.cost_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, leader_agent1, follwer_agent1, follwer_agent2):
        r = torch.tensor(transitions['r_%d' % self.agent_id], dtype=torch.float32)
        c = torch.tensor(transitions['c_%d' % self.agent_id], dtype=torch.float32)
        t = torch.tensor(transitions['t_%d' % self.agent_id], dtype=torch.float32)
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(torch.tensor(transitions['o_%d' % agent_id], dtype=torch.float32))
            u.append(torch.tensor(transitions['u_%d' % agent_id], dtype=torch.float32))
            o_next.append(torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32))

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            u_next_leader1 = self.actor_target_network(o_next[self.agent_id])
            u_next_leader2 = leader_agent1.actor_target_network(o_next[leader_agent1.agent_id])
            u_next_follower1 = follwer_agent1.actor_target_network(torch.cat([o_next[follwer_agent1.agent_id], u_next_leader1, u_next_leader2], dim=1))
            u_next_follower2 = follwer_agent2.actor_target_network(torch.cat([o_next[follwer_agent2.agent_id], u_next_leader1, u_next_leader2], dim=1))
            u_next = [u_next_leader1, u_next_leader2, u_next_follower1, u_next_follower2]

            q_next = self.critic_target_network(o_next[self.agent_id], u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * (1 - t.unsqueeze(1)) * q_next).detach()

            c_next = self.cost_target_network(o_next[self.agent_id], u_next).detach()
            target_c = (c.unsqueeze(1) + self.args.gamma * (1 - t.unsqueeze(1)) * c_next).detach()

        # the q loss
        q_value = self.critic_network(o[self.agent_id], u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the cost loss
        c_value = self.cost_network(o[self.agent_id], u)
        cost_loss = (target_c - c_value).pow(2).mean()

        # the actor loss
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        u[follwer_agent1.agent_id] = follwer_agent1.actor_network(torch.cat([o_next[follwer_agent1.agent_id], u_next_leader1, u_next_leader2], dim=1))

        cost_violation = F.relu(self.cost_network(o[self.agent_id], u) - self.cost_threshold)
        actor_loss = (- self.critic_network(o[self.agent_id], u) + self.l_multiplier * cost_violation).mean()
        # actor_loss = - self.critic_network(o[self.agent_id], u).mean()

        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.cost_optim.zero_grad()
        cost_loss.backward(retain_graph=True)
        self.cost_optim.step()

        # update lagrange multiplier
        # self.l_multiplier += self.args.lr_lagrangian*(self.cost_network(o[self.agent_id], u).mean().item()-self.cost_threshold)
        self.l_multiplier += cost_violation.mean().item() * self.args.lr_lagrangian
        self.l_multiplier = max(0, min(self.l_multiplier, self.args.lagrangian_max_bound))

        if self.train_step > 0 and self.train_step % self.args.update_rate == 0:
            self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    # select action
    def select_action(self, o, noise_rate, epsilon):
        inputs = torch.tensor([o], dtype=torch.float32)
        pi = self.actor_network(inputs)
            # u = pi.cpu().numpy()
            # noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            # u += noise
            # u = np.clip(u, -self.args.high_action, self.args.high_action)
        p = [pi.numpy()[0][0], pi.numpy()[0][1], pi.numpy()[0][2], pi.numpy()[0][3], pi.numpy()[0][4],pi.numpy()[0][5], pi.numpy()[0][6], pi.numpy()[0][7], pi.numpy()[0][8], pi.numpy()[0][9],pi.numpy()[0][10], pi.numpy()[0][11], pi.numpy()[0][12], pi.numpy()[0][13], pi.numpy()[0][14]]
        action_prob = p
        return action_prob
        # return u.copy()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.actor_network.state_dict(), self.model_path + '/' + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/' + 'critic_params.pkl')
        torch.save(self.cost_network.state_dict(), self.model_path + '/' + 'cost_params.pkl')

class Actor_Leader(nn.Module):
    def __init__(self, args, agent_id, input_dim):
        super(Actor_Leader, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)
        # self.action_out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # output = torch.softmax(self.fc3(x), dim=-1)
        softmax = nn.Softmax(dim=1)
        output = softmax(x)
        return output
        # actions = self.max_action * torch.tanh(self.action_out(x))
        #
        # return actions
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear(args.obs_shape[0] + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, state, action, dim=1):
        action = torch.cat(action, dim=dim)
        x = torch.cat([state, action], dim=dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        softmax = nn.Softmax(dim=1)
        output = softmax(x)
        return output

def softmax_action(self, N, M, action_prob):
    mul = [0] * N * (M + 1)
    action = [0] * (M + 1)  # 这里保存单个节点的动作的随机采样的卸载决策的概率值（M+1个）
    action = torch.tensor(action, dtype=torch.float32)  # 这里转化成张量是因为后面的随机采样函数需要是张量的形式
    at = [0] * N  # 保存的是N个结点最终卸载决策
    for i in range(0, N):  # N个结点
        sum_temp = 0.0
        for j in range(0, M + 1):  # M+1个信道增益——提取信道增益
            action[j] = action_prob[i * (M + 1) + j]
            sum_temp = action[j].detach().item() + sum_temp  # 保存的是action中的元素的和  它用于归一化
        for j in range(0, M + 1):  # 将提取出的信道增益归一化
            action[j] = action[j] / sum_temp
            if action_prob[i * (M + 1) + j].detach().item() == 0:
                mul[i * (M + 1) + j] = 0
            else:
                mul[i * (M + 1) + j] = action[j].detach().item() / action_prob[i * (M + 1) + j].detach().item()
        at[i] = Categorical(action).sample().item()
    return at


class Runner:
    def __init__(self, args):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        # set min noise and epsilon
        self.min_noise = args.min_noise_rate
        self.min_epsilon = args.min_epsilon
        # set max episode len
        self.episode_limit = args.max_episode_len
        self.save_path = self.args.save_dir
        # reward or cost record
        self.reward_record = [[] for i in range(N)]
        self.reward_record_leader1 = []
        self.reward_record_leader2 = []
        self.reward_record_follower1 = []
        self.reward_record_follower2 = []
        self.reward_record_bl = []
        self.reward_record_sum = []
        self.follower_action = [[] for i in range(N)]
        self.leader_rate_bl = []
        self.follower_rate_bl = []
        self.leader_rate_sum = []
        self.follower_rate_sum = []
        self.reward_record_sum = []
        self.arrive_record = []
        self.leader_arrive_record = []
        self.follower_arrive_record = []
        self.crash_record = []
        # init agents
        self._init_agents()

    def _init_agents(self):
        self.leader_agent1 = Leader(self.args, 0)

    def run(self, h):
        returns = []
        total_reward = [0, 0, 0, 0, 0]
        done = [False * self.args.n_agents]
        info = None

        with torch.no_grad():
                # choose actions
            leader_actions = []
            leader_action1 = self.leader_agent1.select_action(h, self.noise, self.epsilon)
        return leader_action1

    def update(self, reward):

        reward_tensor = torch.tensor([reward], dtype=torch.float32, requires_grad=True)
        actor_loss = -reward_tensor

        # 这里你可以使用 actor_loss 进行梯度计算和反向传播

        # update the network
        self.leader_agent1.actor_optim.zero_grad()
        actor_loss.backward()
        self.leader_agent1.actor_optim.step()


def fa(a, AA, epsilon, M, t_values, channel_params):
    """
    计算目标函数值（支持15个节点）

    参数:
    a: 优化变量
    AA: 基础值
    epsilon: 常数
    M: M1设备列表
    t_values: t值列表 [t1, t2, ..., t15]
    channel_params: 信道参数列表 [(A1, B1), (A2, B2), ..., (A15, B15)]
    """
    result = AA * a ** (1 / 3)

    for i in range(len(M)):
        t = t_values[i] * (1 - a)
        A, B = channel_params[i]
        numerator = t + A * a
        denominator = t + B * a
        result += epsilon * t * np.log2(numerator / denominator)

    return result


def golden_section_bisection(a, b, M, AA, epsilon, t_values, channel_params, tol=1e-6):
    """
    Golden section search algorithm for finding the minimum of a unimodal function.
    (支持15个节点)
    """
    ratio = (math.sqrt(5) - 1) / 2  # Golden ratio
    X1 = b - ratio * (b - a)
    X2 = a + ratio * (b - a)
    f1 = -fa(X1, AA, epsilon, M, t_values, channel_params)
    f2 = -fa(X2, AA, epsilon, M, t_values, channel_params)

    while abs(b - a) > tol:
        if f1 < f2:
            b = X2
            X2 = X1
            f2 = f1
            X1 = b - ratio * (b - a)
            f1 = -fa(X1, AA, epsilon, M, t_values, channel_params)
        else:
            a = X1
            X1 = X2
            f1 = f2
            X2 = a + ratio * (b - a)
            f2 = -fa(X2, AA, epsilon, M, t_values, channel_params)

    return (X1 + X2) / 2


def get_channel_params(M1_indices, u, p, hi, Hu, Hw):
    """获取信道参数"""
    channel_params = []
    for idx in M1_indices:
        channel_params.append((
            u * p * hi[idx] * Hu[idx],
            u * p * hi[idx] * Hw[idx]
        ))
    return channel_params


def calculate_rates(M1_indices, actions):
    """计算每个M1设备的权重"""
    total_actions = sum(actions[m] for m in M1_indices)
    return [actions[m] / total_actions for m in M1_indices]


def calculate_aa_sum(M0_indices, o, u, p, hi, ki):
    """计算AA（所有M0设备的总和）"""
    return sum((1 / o) * ((u * p * hi[idx]) / ki) ** (1 / 3) for idx in M0_indices)


def calculate_m0_reward(M0_indices, reward, o, u, p, hi, ki, a=1.0):
    """计算M0设备的奖励"""
    for idx in M0_indices:
        reward[idx] = (1 / o) * ((u * p * hi[idx] * a) / ki) ** (1 / 3)


def calculate_m1_reward(M1_indices, reward, epsilon, u, p, hi, Hu, Hw, rates, a):
    """计算M1设备的奖励"""
    for i, idx in enumerate(M1_indices):
        t = rates[i] * (1 - a)
        numerator = u * p * a * hi[idx] * Hu[idx] + t
        denominator = t + u * p * a * hi[idx] * Hw[idx]
        reward[idx] = epsilon * t * np.log2(numerator / denominator)


def rate(M, h, h_list, runner):
    # 常量定义
    o = 700
    B = 10000000
    p = 1.8
    u = 0.7
    Vu = 1.1
    ki = 10 ** -28
    epsilon = B / Vu
    N = 15  # 节点数

    reward = [0] * N
    M0 = np.where(M == 0)[0]
    M1 = np.where(M == 1)[0]
    hi = h[4]
    Hu = h[6]
    Hw = h[7]

    actions = runner.run(h)

    # 只有M0设备的情况
    if len(M0) == N:
        calculate_m0_reward(M0, reward, o, u, p, hi, ki)
    else:
        # 计算AA（所有M0设备的总和）
        AA = calculate_aa_sum(M0, o, u, p, hi, ki)

        if len(M1) > 0:
            # 计算每个M1设备的权重
            rates = calculate_rates(M1, actions)

            # 获取信道参数
            channel_params = get_channel_params(M1, u, p, hi, Hu, Hw)

            # 调用优化函数
            a = golden_section_bisection(0, 1, M1, AA, epsilon, rates, channel_params)

            # 计算奖励
            calculate_m0_reward(M0, reward, o, u, p, hi, ki, a)
            calculate_m1_reward(M1, reward, epsilon, u, p, hi, Hu, Hw, rates, a)
    reward = torch.tensor(reward).unsqueeze(0)
    obj = reward.sum().detach().item()
    return obj


def optimization(h, h_list, M, runner):

    def sum_rate():
        obj = rate(M, h, h_list,  runner)
        return obj
    return sum_rate()


class MemoryDNN:
    def __init__(
            self,
            net,
            learning_rate=0.01,
            training_interval=10,
            batch_size=100,
            memory_size=1000,
            output_graph=False
    ):
        # net: [n_input, n_hidden_1st, n_hidded_2ed, n_output]
        assert (len(net) is 4)  # only 4-layer DNN

        self.net = net
        self.training_interval = training_interval  # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # reset graph
        tf.reset_default_graph()

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

        self.sess = tf.Session()

        # for tensorboard
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('M'):
                w3 = tf.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # ------------------ build memory_net ------------------
        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')  # input
        self.m = tf.placeholder(tf.float32, [None, self.net[-1]], name='mode')  # for calculating loss
        self.is_train = tf.placeholder("bool")  # train or evaluate

        with tf.variable_scope('memory_net'):
            c_names, w_initializer, b_initializer = \
                ['memory_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                    tf.random_normal_initializer(0., 1 / self.net[0]), tf.constant_initializer(0.1)  # config of layers

            self.m_pred = build_layers(self.h, c_names, self.net, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m, logits=self.m_pred))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr, 0.09).minimize(self.loss)

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
        #        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, 0: self.net[0]]
        m_train = batch_memory[:, self.net[0]:]
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.h: h_train, self.m: m_train})

        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):
        m_pred = self.sess.run(self.m_pred, feed_dict={self.h: h})

        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k=1):
        # return k-nearest-mode
        m_list = []
        m_list.append(1 * (m > 0))
        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0:
                    # set a positive user to 0
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    # set a negtive user to 1
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k=1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


class PolicyNetwork(nn.Module):
   """策略网络，输出卸载决策的概率分布"""

   def __init__(self):
       super(PolicyNetwork, self).__init__()
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2, stride=1, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2),
       )
       self.flatten = nn.Flatten()

       # 全连接层
       self.fc_layers = nn.Sequential(
           nn.Linear(270, 300),
           nn.Tanh(),
           nn.Linear(300, 450),
           nn.Tanh(),
           nn.Linear(450, 300),
           nn.Tanh(),
           nn.Linear(300, 120),
           nn.Tanh(),
           nn.Linear(120, 100),
           nn.Tanh(),
       )
       self.output_layer = nn.Linear(100, N * (M + 1))
       self.softmax = nn.Softmax(dim=1)

       # 权重初始化
       self._initialize_weights()

   def _initialize_weights(self):
       for layer in [self.fc_layers, self.output_layer]:
           if hasattr(layer, 'weight'):
               layer.weight.data.normal_(0, 0.01)

   def forward(self, x):
       x = x.view(-1, N * M)
       x = torch.unsqueeze(x, dim=1)
       x = torch.unsqueeze(x, dim=1)
       x = self.conv1(x)
       x = self.flatten(x)
       x = self.fc_layers(x)
       output = self.output_layer(x)
       return self.softmax(output)


class ACPolicyNetwork(nn.Module):
   """AC算法的策略网络"""

   def __init__(self):
       super(ACPolicyNetwork, self).__init__()
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2, stride=1, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2),
       )
       self.flatten = nn.Flatten()

       # 更深的网络结构
       self.fc_layers = nn.Sequential(
           nn.Linear(270, 300),
           nn.Sigmoid(),
           nn.Linear(300, 450),
           nn.Sigmoid(),
           nn.Linear(450, 600),
           nn.Sigmoid(),
           nn.Linear(600, 700),
           nn.Sigmoid(),
           nn.Linear(700, 800),
           nn.Sigmoid(),
       )

       self.output_layer = nn.Linear(800, N * (M + 1))
       self.softmax = nn.Softmax(dim=1)

       # 权重初始化
       self._initialize_weights()

   def _initialize_weights(self):
       for layer in [self.fc_layers, self.output_layer]:
           if hasattr(layer, 'weight'):
               layer.weight.data.normal_(0, 0.01)

   def forward(self, x):
       x = x.view(-1, N * M)
       x = torch.unsqueeze(x, dim=1)
       x = torch.unsqueeze(x, dim=1)
       x = self.conv1(x)
       x = self.flatten(x)
       x = self.fc_layers(x)
       output = self.output_layer(x)
       return self.softmax(output)


class CriticNetwork(nn.Module):
   """评论家网络"""

   def __init__(self):
       super(CriticNetwork, self).__init__()
       self.fc_layers = nn.Sequential(
           nn.Linear(N * M, 120),
           nn.ReLU(),
           nn.Linear(120, 250),
           nn.ReLU(),
           nn.Linear(250, 400),
           nn.ReLU(),
           nn.Linear(400, 550),
           nn.ReLU(),
           nn.Linear(550, 700),
           nn.ReLU(),
       )
       self.output_layer = nn.Linear(700, 1)

       # 权重初始化
       self._initialize_weights()

   def _initialize_weights(self):
       for layer in [self.fc_layers, self.output_layer]:
           if hasattr(layer, 'weight'):
               layer.weight.data.normal_(0, 0.01)

   def forward(self, x):
       x = x.view(-1, N * M)
       x = self.fc_layers(x)
       return self.output_layer(x)


# DDPG Networks
class DDPGActor(nn.Module):
   def __init__(self):
       super(DDPGActor, self).__init__()
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=15, kernel_size=2, stride=1, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2)
       )
       self.flatten = nn.Flatten()
       self.fc1 = nn.Linear(270, 300)
       self.fc2 = nn.Linear(300, 450)
       self.fc3 = nn.Linear(450, 600)
       self.fc4 = nn.Linear(600, 700)
       self.fc5 = nn.Linear(700, 800)
       self.out = nn.Linear(800, ACTION_DIM)

       # Initialize weights
       for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.out]:
           layer.weight.data.normal_(0, 0.01)

       self.sigmoid = nn.Sigmoid()

   def forward(self, state):
       state = state.view(-1, 1, 1, STATE_DIM)
       x = self.conv1(state)
       x = self.flatten(x)
       x = F.sigmoid(self.fc1(x))
       x = F.sigmoid(self.fc2(x))
       x = F.sigmoid(self.fc3(x))
       x = F.sigmoid(self.fc4(x))
       x = F.sigmoid(self.fc5(x))
       actions = self.sigmoid(self.out(x)) * ACTION_BOUND
       return actions


class DDPGCritic(nn.Module):
   def __init__(self):
       super(DDPGCritic, self).__init__()
       self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, 120)
       self.fc2 = nn.Linear(120, 250)
       self.fc3 = nn.Linear(250, 400)
       self.fc4 = nn.Linear(400, 550)
       self.fc5 = nn.Linear(550, 700)
       self.out = nn.Linear(700, 1)

       # Initialize weights
       for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.out]:
           layer.weight.data.normal_(0, 0.01)

   def forward(self, state, action):
       x = torch.cat([state, action], 1)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.relu(self.fc3(x))
       x = F.relu(self.fc4(x))
       x = F.relu(self.fc5(x))
       q_value = self.out(x)
       return q_value


class PolicyLearner:
   """策略学习器 - 你的原始算法"""

   def __init__(self, learning_rate=0.00001):
       self.policy_net = PolicyNetwork()
       self.baseline_net = PolicyNetwork()
       self.optimizer_policy = torch.optim.AdamW(
           self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-2
       )
       self.optimizer_baseline = torch.optim.AdamW(
           self.baseline_net.parameters(), lr=learning_rate, weight_decay=1e-2
       )
       self.loss_history = []

   def get_action_probability(self, state, use_baseline=False):
       """获取动作概率分布"""
       state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
       network = self.baseline_net if use_baseline else self.policy_net
       return network(state_tensor)

   def compute_joint_probability(self, action_prob, actions):
       """计算联合概率"""
       joint_prob = 1.0
       for i in range(N):
           joint_prob *= action_prob[0][(M + 1) * i + actions[i]]
       return joint_prob

   def update_policy(self, reward, baseline, action_prob, actions, step_count):
       """更新策略网络"""
       joint_prob = self.compute_joint_probability(action_prob, actions)
       policy_loss = -(reward - baseline) * torch.log(joint_prob)
       detail_loss = -reward * torch.log(joint_prob)

       self.optimizer_policy.zero_grad()
       policy_loss.backward()
       self.loss_history.append(detail_loss.detach().item())
       self.optimizer_policy.step()

       # 定期同步基线网络
       if step_count % 500 == 0:
           self.optimizer_baseline.step()


class PGPolicyLearner:
   """PG算法学习器 - 对比算法"""

   def __init__(self, learning_rate=0.0001):
       self.policy_net = PolicyNetwork()
       self.baseline_net = PolicyNetwork()
       self.optimizer_policy = torch.optim.AdamW(
           self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-2
       )
       self.optimizer_baseline = torch.optim.AdamW(
           self.baseline_net.parameters(), lr=learning_rate, weight_decay=1e-2
       )
       self.loss_history = []

   def get_action_probability(self, state, use_baseline=False):
       """获取动作概率分布"""
       state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
       network = self.baseline_net if use_baseline else self.policy_net
       return network(state_tensor)

   def compute_joint_probability(self, action_prob, actions):
       """计算联合概率"""
       joint_prob = 1.0
       for i in range(N):
           joint_prob *= action_prob[0][(M + 1) * i + actions[i]]
       return joint_prob

   def update_policy(self, reward, baseline, action_prob, actions, step_count):
       """更新策略网络 - PG版本（无基线）"""
       baseline = 0  # PG算法中基线设为0
       joint_prob = self.compute_joint_probability(action_prob, actions)
       policy_loss = -(reward - baseline) * torch.log(joint_prob)
       detail_loss = -reward * torch.log(joint_prob)

       self.optimizer_policy.zero_grad()
       policy_loss.backward()
       self.loss_history.append(detail_loss.detach().item())
       self.optimizer_policy.step()


class ACActorCriticLearner:
   """AC算法学习器"""

   def __init__(self, policy_lr=0.00001, critic_lr=0.0001, gamma=0.9):
       self.policy_net = ACPolicyNetwork()
       self.critic_net = CriticNetwork()
       self.optimizer_policy = torch.optim.AdamW(
           self.policy_net.parameters(), lr=policy_lr, weight_decay=1e-2
       )
       self.optimizer_critic = torch.optim.AdamW(
           self.critic_net.parameters(), lr=critic_lr
       )
       self.loss_fn = torch.nn.MSELoss()
       self.gamma = gamma
       self.loss_history = []

   def get_action_probability(self, state):
       """获取动作概率分布"""
       state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
       return self.policy_net(state_tensor)

   def get_state_value(self, state):
       """获取状态价值"""
       state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
       return self.critic_net(state_tensor)

   def compute_joint_probability(self, action_prob, actions):
       """计算联合概率"""
       joint_prob = 1.0
       for i in range(N):
           joint_prob *= action_prob[0][(M + 1) * i + actions[i]]
       return joint_prob

   def update_policy(self, reward, baseline, action_prob, actions, current_value, next_value):
       """更新策略网络"""
       td_target = reward + self.gamma * next_value
       td_delta = td_target - current_value
       joint_prob = self.compute_joint_probability(action_prob, actions)

       loss_policy = torch.mean(-torch.log(joint_prob) * td_delta.detach() / 3)

       self.optimizer_policy.zero_grad()
       loss_policy.backward()
       self.optimizer_policy.step()

       self.loss_history.append(loss_policy.detach().item())

   def update_critic(self, reward, current_value, next_value):
       """更新评论家网络"""
       td_target = reward + self.gamma * next_value
       loss_critic = self.loss_fn(current_value, td_target.detach())

       self.optimizer_critic.zero_grad()
       loss_critic.backward()
       self.optimizer_critic.step()


class DDPGLearner:
   """DDPG算法学习器"""

   def __init__(self):
       self.actor = DDPGActor().to(device)
       self.target_actor = DDPGActor().to(device)
       self.critic = DDPGCritic().to(device)
       self.target_critic = DDPGCritic().to(device)

       # Initialize target networks with same weights
       self.target_actor.load_state_dict(self.actor.state_dict())
       self.target_critic.load_state_dict(self.critic.state_dict())

       # Optimizers
       self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
       self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

       # Memory buffer
       self.memory = deque(maxlen=MEMORY_CAPACITY)
       self.loss_fn = nn.MSELoss()
       self.loss_history = []

   def get_action(self, state, add_noise=True):
       state = torch.FloatTensor(state).unsqueeze(0).to(device)
       action = self.actor(state).cpu().data.numpy().flatten()

       if add_noise:
           noise = np.random.normal(0, EXPLORE_NOISE, size=action.shape)
           action = np.clip(action + noise, 0, ACTION_BOUND)

       return action

   def remember(self, state, action, reward, next_state, done):
       self.memory.append((state, action, reward, next_state, done))

   def learn(self):
       if len(self.memory) < BATCH_SIZE:
           return 0

       # Sample a batch from memory
       batch = random.sample(self.memory, BATCH_SIZE)
       states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
       actions = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
       rewards = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(device)
       next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
       dones = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1).to(device)

       # Update critic
       next_actions = self.target_actor(next_states)
       target_q = rewards + (1 - dones) * GAMMA * self.target_critic(next_states, next_actions)
       current_q = self.critic(states, actions)
       critic_loss = self.loss_fn(current_q, target_q.detach())

       self.critic_optimizer.zero_grad()
       critic_loss.backward()
       self.critic_optimizer.step()

       # Update actor
       actor_loss = -self.critic(states, self.actor(states)).mean()

       self.actor_optimizer.zero_grad()
       actor_loss.backward()
       self.actor_optimizer.step()

       # Soft update target networks
       self.soft_update(self.actor, self.target_actor)
       self.soft_update(self.critic, self.target_critic)

       total_loss = critic_loss.item() + actor_loss.item()
       self.loss_history.append(total_loss)

       return total_loss

   def soft_update(self, local_model, target_model):
       for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
           target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)


class DROOLearner:
   """DROO算法学习器"""

   def __init__(self, memory_size=DROO_MEMORY_SIZE, delta=DROO_DELTA):
       self.memory_size = memory_size
       self.delta = delta
       self.K = 1  # 初始K值
       self.decoder_mode = 'OP'  # Order-preserving decoding

       # 初始化DROO记忆网络
       self.mem = MemoryDNN(
           net=[N, 120, 80, N],
           learning_rate=0.0001,
           training_interval=10,
           batch_size=128,
           memory_size=memory_size
       )

       self.reward_history = []
       self.k_history = []
       self.loss_history = []

   def get_action(self, state):
       """DROO动作选择"""
       # 生成K个候选动作
       m_list = self.mem.decode(state, self.K, self.decoder_mode)

       # 评估每个动作的奖励
       r_list = []
       for m in m_list:
           reward = optimization(state, 0, m, self.runner)
           r_list.append(reward )

       # 选择最佳动作
       best_action = m_list[np.argmax(r_list)]
       best_reward = np.max(r_list)

       # 编码最佳动作到记忆中
       self.mem.encode(state[0], best_action)

       # 自适应调整K
       if len(self.reward_history) > 0 and len(self.reward_history) % self.delta == 0:
           recent_k = max(self.k_history[-self.delta:]) + 1
           self.K = min(recent_k + 1, N)

       # 记录历史
       self.reward_history.append(best_reward)
       self.k_history.append(np.argmax(r_list))

       return best_action, best_reward

   def progress_channel(self, decision, h):
       """处理信道增益（与DROO原代码一致）"""
       M1 = np.where(decision == 1)[0]
       h_list = [[0] * N for _ in range(7)]

       for de in range(len(decision)):
           if de in M1:
               for i in range(7):
                   h_list[i][de] = h[i][de]
           else:
               for i in range(7):
                   h_list[i][de] = 0
       return h_list

   def set_runner(self, runner):
       """设置运行器"""
       self.runner = runner


# H2AC Networks
class H2ACPolicyNetwork(nn.Module):
    def __init__(self):
        super(H2ACPolicyNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(675, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, N * (M + 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 8, 15)  # [batch_size, 8, 5]
        x = x.unsqueeze(1)  # [batch_size, 1, 8, 5]
        x = self.conv1(x)  # [batch_size, 15, 5, 3]
        x = self.flatten(x)  # [batch_size, 15*5*3=225]
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.out(x)
        return self.softmax(x)


class H2ACContinuousActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(H2ACContinuousActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

        self.mean = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)

        self.log_std = nn.Parameter(torch.zeros(output_dim) - 1.0)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, state):
        if torch.isnan(state).any():
            raise ValueError("NaN values in state input")

        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        mean = 0.5 * torch.tanh(self.mean(x)) + 0.5  # Output in [0,1]
        log_std = torch.clamp(self.log_std, min=-5, max=2)
        std = torch.exp(log_std) + 1e-6

        if torch.isnan(mean).any() or torch.isnan(std).any():
            raise ValueError("NaN values in network output")

        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return action, log_prob


class H2ACCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(H2ACCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class H2ACLearner:
    """H2AC算法学习器 - 混合离散连续动作空间"""

    def __init__(self):
        self.policy_net = H2ACPolicyNetwork()
        self.continuous_actor = H2ACContinuousActor(120, 15)
        self.critic_net = H2ACCriticNetwork(120, 15)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=H2AC_LR_POLICY)
        self.continuous_actor_optimizer = torch.optim.Adam(self.continuous_actor.parameters(), lr=H2AC_LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=H2AC_LR_CRITIC)

        self.replay_buffer = deque(maxlen=1000)
        self.reward_history = []
        self.loss_history = []

    def get_discrete_action(self, state):
        """获取离散动作概率分布"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
        return action_probs[0]

    def get_continuous_action(self, state):
        """获取连续动作"""
        state_tensor = torch.FloatTensor(state.flatten())
        with torch.no_grad():
            action, _ = self.continuous_actor.sample(state_tensor)
        return action

    def process_channel(self, decision, h):
        """处理信道增益"""
        M1_indices = np.where(decision == 1)[0]
        M0_indices = np.where(decision == 0)[0]

        processed_channels = [[0] * N for _ in range(8)]

        for idx, de in enumerate(M1_indices):
            for i in range(8):
                processed_channels[i][idx] = h[i][de]

        for idx, dex in enumerate(M0_indices):
            de = idx + len(M1_indices)
            for i in range(8):
                processed_channels[i][de] = 0

        return processed_channels

    def update(self, batch_size=H2AC_BATCH_SIZE):
        """更新网络参数"""
        if len(self.replay_buffer) < batch_size:
            return 0

        batch = random.sample(self.replay_buffer, batch_size)
        states, old_probs, discrete_actions, cont_actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        cont_actions = torch.stack(cont_actions)
        discrete_actions = torch.stack(discrete_actions)
        old_probs = torch.stack(old_probs)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Flatten states
        flat_states = states.view(batch_size, -1)
        flat_next_states = next_states.view(batch_size, -1)

        total_loss = 0

        # Critic update
        with torch.no_grad():
            next_cont_actions, _ = self.continuous_actor.sample(flat_next_states)
            next_q_values = self.critic_net(flat_next_states, next_cont_actions)
            td_target = rewards + (1 - dones) * H2AC_GAMMA * next_q_values

        current_q_values = self.critic_net(flat_states, cont_actions)
        critic_loss = F.mse_loss(current_q_values, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        total_loss += critic_loss.item()

        # Continuous actor update
        new_cont_actions, log_probs = self.continuous_actor.sample(flat_states)
        q_values = self.critic_net(flat_states, new_cont_actions)
        cont_actor_loss = -(q_values).mean()

        self.continuous_actor_optimizer.zero_grad()
        cont_actor_loss.backward()
        self.continuous_actor_optimizer.step()
        total_loss += cont_actor_loss.item()

        # Discrete actor update
        new_probs = self.policy_net(flat_states)
        new_probs = new_probs.view(batch_size, N, M + 1)
        new_probs = new_probs.view(-1, M + 1)
        discrete_actions = discrete_actions.view(-1).long()

        dist = Categorical(new_probs)
        log_probs = dist.log_prob(discrete_actions)

        with torch.no_grad():
            advantage = (td_target - current_q_values).repeat(1, N).view(-1)

        policy_loss = -(log_probs * advantage).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        total_loss += policy_loss.item()

        self.loss_history.append(total_loss)
        return total_loss

    def set_runner(self, runner):
        """设置运行器"""
        self.runner = runner

class ChannelProcessor:
   """信道处理器"""

   @staticmethod
   def process_channel(decisions, channel_gains):
       """处理信道增益"""
       M1_indices = np.where(decisions == 1)[0]
       M0_indices = np.where(decisions == 0)[0]

       processed_channels = [[0] * N for _ in range(8)]

       for idx, de in enumerate(M1_indices):
           for i in range(8):
               processed_channels[i][idx] = channel_gains[i][de]

       for idx, dex in enumerate(M0_indices):
           de = idx + len(M1_indices)
           for i in range(8):
               processed_channels[i][de] = 0

       return processed_channels


class ResultComparator:
   """结果比较器"""

   def __init__(self):
       self.results = defaultdict(list)

   def add_result(self, algorithm_name, rewards, losses=None, training_time=None, k_history=None):
       """添加算法结果"""
       self.results[algorithm_name] = {
           'rewards': rewards,
           'losses': losses if losses else [],
           'training_time': training_time,
           'k_history': k_history,
           'mean_reward': np.mean(rewards),
           'std_reward': np.std(rewards),
           'max_reward': np.max(rewards),
           'min_reward': np.min(rewards),
           'final_reward': rewards[-1] if rewards else 0,
           'convergence_step': self._find_convergence_step(rewards) if rewards else 0
       }

   def _find_convergence_step(self, rewards, window=100, threshold=0.95):
       """找到收敛步数"""
       if len(rewards) < window:
           return len(rewards)

       max_reward = np.max(rewards)
       for i in range(len(rewards) - window):
           if np.mean(rewards[i:i + window]) >= threshold * max_reward:
               return i
       return len(rewards)

   def generate_comparison_table(self):
       """生成对比表格"""
       comparison_data = []
       for algo_name, result in self.results.items():
           comparison_data.append({
               'Algorithm': algo_name,
               'Mean Reward': f"{result['mean_reward']:.4f}",
               'Std Reward': f"{result['std_reward']:.4f}",
               'Max Reward': f"{result['max_reward']:.4f}",
               'Min Reward': f"{result['min_reward']:.4f}",
               'Final Reward': f"{result['final_reward']:.4f}",
               'Convergence Step': result['convergence_step'],
               'Training Time (s)': f"{result['training_time']:.2f}" if result['training_time'] else 'N/A'
           })

       df = pd.DataFrame(comparison_data)
       return df

   def plot_comparison(self, save_path=None):
       """绘制对比图表"""
       fig, (ax1) = plt.subplots(1, figsize=(16, 6))

       # 绘制奖励曲线
       for algo_name, result in self.results.items():
           rewards = result['rewards']
           smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
           ax1.plot(smoothed_rewards, label=algo_name, alpha=0.8, linewidth=2)

       ax1.set_xlabel('Training Steps')
       ax1.set_ylabel('Reward')
       ax1.set_title('Reward Comparison (Smoothed)')
       ax1.legend()
       ax1.grid(True, alpha=0.3)

       # # 绘制损失曲线（如果存在）
       # has_losses = any(result['losses'] for result in self.results.values())
       # if has_losses:
       #     for algo_name, result in self.results.items():
       #         if result['losses']:
       #             losses = result['losses']
       #             smoothed_losses = pd.Series(losses).rolling(window=100, min_periods=1).mean()
       #             ax2.plot(smoothed_losses, label=algo_name, alpha=0.8, linewidth=2)
       #
       #     ax2.set_xlabel('Training Steps')
       #     ax2.set_ylabel('Loss')
       #     ax2.set_title('Loss Comparison (Smoothed)')
       #     ax2.legend()
       #     ax2.grid(True, alpha=0.3)

       plt.tight_layout()
       if save_path:
           plt.savefig(save_path, dpi=200, bbox_inches='tight')
       plt.show()

   def save_detailed_results(self, filename='detailed_results.csv'):
       """保存详细结果"""
       detailed_data = []
       max_length = max(len(result['rewards']) for result in self.results.values())

       for step in range(max_length):
           row = {'Step': step}
           for algo_name, result in self.results.items():
               if step < len(result['rewards']):
                   row[f'{algo_name}_Reward'] = result['rewards'][step]
               else:
                   row[f'{algo_name}_Reward'] = np.nan
           detailed_data.append(row)

       df = pd.DataFrame(detailed_data)
       df.to_csv(filename, index=False)
       print(f"详细结果已保存到 {filename}")


class AlgorithmRunner:
   """算法运行器基类"""

   def __init__(self, learner_class, algorithm_name):
       self.learner = learner_class()
       self.algorithm_name = algorithm_name
       self.channel_processor = ChannelProcessor()
       self.reward_history = []
       self.loss_history = []

       # 初始化环境
       args = get_args()
       args.n_players = args.n_agents = args.n_leader = 1
       args.obs_shape = [120]
       args.action_shape = [15]
       args.terminal_shape = [1]
       args.high_action, args.low_action = 1, 0
       self.runner = Runner(args)

   def normalize_channel_gains(self, channel_gains):
       """归一化信道增益"""
       total_gain = sum(channel_gains)
       return [gain / total_gain for gain in channel_gains]

   def _get_normalized_action_distribution(self, action_prob, node_index):
       """获取归一化的动作分布"""
       start_idx = node_index * (M + 1)
       end_idx = start_idx + (M + 1)
       action_slice = action_prob[0][start_idx:end_idx]

       # 归一化
       sum_temp = action_slice.sum()
       return action_slice / sum_temp if sum_temp > 0 else action_slice

   def sample_actions(self, action_prob, num_samples=1):
       """采样动作"""
       actions_list = []
       for _ in range(num_samples):
           actions = []
           for i in range(N):
               action_dist = self._get_normalized_action_distribution(action_prob, i)
               actions.append(Categorical(action_dist).sample().item())
           actions_list.append(actions)
       return actions_list

   def run(self, channel_data, num_steps=10000):
       """运行算法"""
       raise NotImplementedError("子类必须实现此方法")


# 原有的算法运行器类保持不变（OriginalAlgorithmRunner, PGAlgorithmRunner, ACAlgorithmRunner, DDPGAlgorithmRunner）
# 这里省略以节省空间，实际使用时需要包含

class DROOAlgorithmRunner(AlgorithmRunner):
   """DROO算法运行器"""

   def __init__(self):
       super().__init__(DROOLearner, "DROO")
       self.learner.set_runner(self.runner)
       self.k_history = []

   def run(self, channel_data, num_steps=10000):
       start_time = time.time()

       for step in range(num_steps):
           # 获取当前信道状态
           current_channels = channel_data[step]

           # DROO动作选择
           action, reward = self.learner.get_action(current_channels)

           # 记录奖励和K值
           self.reward_history.append(reward)
           self.k_history.append(self.learner.K)

           # 定期训练DNN
           if step % self.learner.mem.training_interval == 0:
               self.learner.mem.learn()
               if hasattr(self.learner.mem, 'cost_his') and self.learner.mem.cost_his:
                   self.loss_history.append(self.learner.mem.cost_his[-1])

           # 进度显示
           if step % 1000 == 0:
               progress = (step / num_steps) * 100
               current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                   self.reward_history)
               print(
                   f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}, Current K: {self.learner.K}")

       total_time = time.time() - start_time
       print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

       return self.reward_history, self.loss_history, total_time
class OriginalAlgorithmRunner(AlgorithmRunner):
   """原始算法运行器"""

   def __init__(self):
       super().__init__(PolicyLearner, "OURs")

   def run(self, channel_data, num_steps=10000):
       start_time = time.time()

       for step in range(num_steps):
           # 获取当前信道状态
           current_channels = channel_data[step]
           normalized_channels = self.normalize_channel_gains(current_channels)

           # 策略网络动作选择
           policy_probs = self.learner.get_action_probability(normalized_channels)
           action_candidates = self.sample_actions(policy_probs, num_samples=5)

           # 寻找最佳动作
           best_reward = -float('inf')
           best_actions = None

           for actions in action_candidates:
               processed_channels = self.channel_processor.process_channel(
                   np.array(actions), current_channels
               )
               reward = optimization(
                   current_channels, processed_channels, np.array(actions), self.runner
               )
               reward_value = reward
               if reward_value > best_reward:
                   best_reward = reward_value
                   best_actions = actions

           # 基线网络评估
           baseline_probs = self.learner.get_action_probability(
               normalized_channels, use_baseline=True
           )
           baseline_actions = self.sample_actions(baseline_probs)[0]

           processed_baseline = self.channel_processor.process_channel(
               np.array(baseline_actions), current_channels
           )
           baseline_reward = optimization(
               current_channels, processed_baseline, np.array(baseline_actions), self.runner
           )
           baseline_value = baseline_reward

           # 记录奖励
           self.reward_history.append(best_reward)

           # 更新策略
           self.learner.update_policy(
               best_reward, baseline_value, policy_probs, best_actions, step
           )

           # 进度显示
           if step % 1000 == 0:
               progress = (step / num_steps) * 100
               print(f"{self.algorithm_name} Progress: {progress:.1f}%")

       total_time = time.time() - start_time
       print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

       return self.reward_history, self.learner.loss_history, total_time


class PGAlgorithmRunner(AlgorithmRunner):
   """PG算法运行器"""

   def __init__(self):
       super().__init__(PGPolicyLearner, "PG")

   def run(self, channel_data, num_steps=10000):
       start_time = time.time()

       for step in range(num_steps):
           # 获取当前信道状态
           current_channels = channel_data[step]
           normalized_channels = self.normalize_channel_gains(current_channels)

           # 策略网络动作选择
           policy_probs = self.learner.get_action_probability(current_channels)
           actions = self.sample_actions(policy_probs)[0]

           # 计算奖励
           processed_channels = self.channel_processor.process_channel(
               np.array(actions), current_channels
           )
           reward = optimization(
               current_channels, processed_channels, np.array(actions), self.runner
           )
           reward_value = reward

           # 记录奖励
           self.reward_history.append(reward_value)

           # 更新策略（PG版本，基线设为0）
           self.learner.update_policy(reward_value, 0, policy_probs, actions, step)

           # 进度显示
           if step % 1000 == 0:
               progress = (step / num_steps) * 100
               print(f"{self.algorithm_name} Progress: {progress:.1f}%")

       total_time = time.time() - start_time
       print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

       return self.reward_history, self.learner.loss_history, total_time


class ACAlgorithmRunner(AlgorithmRunner):
   """AC算法运行器"""

   def __init__(self):
       super().__init__(ACActorCriticLearner, "AC")

   def run(self, channel_data, num_steps=10000):
       start_time = time.time()

       for step in range(num_steps):
           # 获取当前信道状态
           current_channels = channel_data[step]

           # 策略网络动作选择
           policy_probs = self.learner.get_action_probability(current_channels)
           actions = self.sample_actions(policy_probs)[0]

           # 计算奖励
           processed_channels = self.channel_processor.process_channel(
               np.array(actions), current_channels
           )
           reward = optimization(
               current_channels, processed_channels, np.array(actions), self.runner
           )
           reward_value = reward 

           # 记录奖励
           self.reward_history.append(reward_value)

           # 获取当前状态价值
           current_value = self.learner.get_state_value(current_channels)

           # 获取下一状态价值（如果是最后一步，则为0）
           if step < num_steps - 1:
               next_channels = channel_data[step + 1]
               next_value = self.learner.get_state_value(next_channels)
           else:
               next_value = torch.tensor(0.0)

           # 更新策略和评论家
           self.learner.update_policy(
               reward_value, 0, policy_probs, actions, current_value, next_value
           )
           self.learner.update_critic(reward_value, current_value, next_value)

           # 进度显示
           if step % 1000 == 0:
               progress = (step / num_steps) * 100
               current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                   self.reward_history)
               print(f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}")

       total_time = time.time() - start_time
       print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

       return self.reward_history, self.learner.loss_history, total_time


class DDPGAlgorithmRunner(AlgorithmRunner):
   """DDPG算法运行器"""

   def __init__(self):
       super().__init__(DDPGLearner, "DDPG")

   def run(self, channel_data, num_steps=10000):
       start_time = time.time()

       for step in range(num_steps):
           # 获取当前信道状态
           current_channels = channel_data[step]

           # 获取动作
           action = self.learner.get_action(current_channels[3])  # 使用第3个信道

           # 将连续动作转换为离散决策
           decisions = (action > 0.5).astype(int)

           # 计算奖励
           processed_channels = self.channel_processor.process_channel(
               decisions, current_channels
           )
           reward = optimization(
               current_channels, processed_channels, decisions, self.runner
           )
           reward_value = reward 

           # 获取下一状态
           if step < num_steps - 1:
               next_state = channel_data[step + 1][3]
           else:
               next_state = current_channels[3]

           # 存储经验
           self.learner.remember(current_channels[3], action, reward_value, next_state, True)

           # 学习
           loss = self.learner.learn()
           if loss > 0:
               self.loss_history.append(loss)

           # 记录奖励
           self.reward_history.append(reward_value)

           # 进度显示
           if step % 1000 == 0:
               progress = (step / num_steps) * 100
               current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                   self.reward_history)
               print(f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}")

       total_time = time.time() - start_time
       print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

       return self.reward_history, self.loss_history, total_time


class H2ACAlgorithmRunner(AlgorithmRunner):
    """H2AC算法运行器"""

    def __init__(self):
        super().__init__(H2ACLearner, "H2AC")
        self.learner.set_runner(self.runner)

    def run(self, channel_data, num_steps=10000):
        start_time = time.time()

        for step in range(num_steps):
            # 获取当前信道状态
            current_channels = channel_data[step]

            # 获取离散和连续动作
            discrete_action_probs = self.learner.get_discrete_action(current_channels)
            continuous_action = self.learner.get_continuous_action(current_channels)

            # 处理离散动作
            decisions = []
            for i in range(N):
                probs = discrete_action_probs[i * (M + 1):(i + 1) * (M + 1)]
                probs = probs / probs.sum()
                decisions.append(Categorical(torch.FloatTensor(probs)).sample().item())
            decisions = np.array(decisions)

            # 计算奖励
            processed_channels = self.learner.process_channel(decisions, current_channels)
            reward = optimization(
                current_channels, processed_channels, decisions, self.runner
            )
            reward_value = reward 

            # 获取下一状态
            if step < num_steps - 1:
                next_state = channel_data[step + 1]
            else:
                next_state = current_channels

            # 存储经验
            self.learner.replay_buffer.append((
                torch.FloatTensor(current_channels),
                discrete_action_probs.detach(),
                torch.FloatTensor(decisions),
                continuous_action.detach(),
                reward_value,
                torch.FloatTensor(next_state),
                True if step == num_steps - 1 else False
            ))

            # 记录奖励
            self.reward_history.append(reward_value)

            # 更新网络
            loss = self.learner.update()
            if loss > 0:
                self.loss_history.append(loss)

            # 进度显示
            if step % 1000 == 0:
                progress = (step / num_steps) * 100
                current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                    self.reward_history)
                print(f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}")

        total_time = time.time() - start_time
        print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

        return self.reward_history, self.loss_history, total_time


# A3C Networks
class A3CActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Actor 部分
        self.fc1 = nn.Linear(STATE_DIM_A3C, 300)
        self.fc2 = nn.Linear(300, 450)
        self.fc3 = nn.Linear(450, 300)
        self.fc4 = nn.Linear(300, 120)
        self.fc5 = nn.Linear(120, 100)
        self.actor_out = nn.Linear(100, JOINT_ACTION_DIM)

        # Critic 部分
        self.critic_fc1 = nn.Linear(STATE_DIM_A3C, 120)
        self.critic_fc2 = nn.Linear(120, 250)
        self.critic_fc3 = nn.Linear(250, 400)
        self.critic_fc4 = nn.Linear(400, 550)
        self.critic_fc5 = nn.Linear(550, 700)
        self.critic_out = nn.Linear(700, 1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_actor(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        logits = self.actor_out(x)
        logits = logits.view(-1, N, ACTION_PER_NODE)
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_critic(self, s):
        x = F.relu(self.critic_fc1(s))
        x = F.relu(self.critic_fc2(x))
        x = F.relu(self.critic_fc3(x))
        x = F.relu(self.critic_fc4(x))
        x = F.relu(self.critic_fc5(x))
        v = self.critic_out(x)
        return v.squeeze(-1)


class A3CLearner:
    """A3C算法学习器 - 异步优势Actor-Critic"""

    def __init__(self):
        self.model = A3CActorCritic().to(device)
        self.optimizer_actor = torch.optim.Adam(self.model.parameters(), lr=A3C_LR_ACTOR)
        self.optimizer_critic = torch.optim.Adam(self.model.parameters(), lr=A3C_LR_CRITIC)
        self.reward_history = []
        self.loss_history = []

    def select_action(self, state):
        """选择动作"""
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        probs = self.model.forward_actor(s).squeeze(0)  # (N, A)
        actions = []
        logp_terms = []

        for i in range(N):
            dist = Categorical(probs[i])
            a = dist.sample()
            actions.append(a.item())
            logp_terms.append(dist.log_prob(a))

        logp = torch.stack(logp_terms).sum()
        value = self.model.forward_critic(s).squeeze(0)

        return actions, logp, value

    def update(self, trajectory):
        """更新网络参数"""
        if not trajectory:
            return 0

        R = 0
        actor_loss = 0
        critic_loss = 0
        returns = []

        # 计算折扣回报
        for (_, _, _, reward, done, _) in reversed(trajectory):
            R = reward + A3C_GAMMA * R * (1 - done)
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        for (state, action, logp, reward, done, value), R in zip(trajectory, returns):
            advantage = R - value
            # Critic loss
            critic_loss += advantage.pow(2)
            # Actor loss (带熵正则化)
            actor_loss += -logp * advantage.detach() - A3C_ENTROPY_COEF * logp

        # 合并损失
        total_loss = actor_loss + critic_loss

        # 更新网络
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), A3C_MAX_GRAD_NORM)

        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.loss_history.append(total_loss.item())
        return total_loss.item()

    def set_runner(self, runner):
        """设置运行器"""
        self.runner = runner


class A3CAlgorithmRunner(AlgorithmRunner):
    """A3C算法运行器"""

    def __init__(self):
        super().__init__(A3CLearner, "A3C")
        self.learner.set_runner(self.runner)

    def run(self, channel_data, num_steps=10000):
        start_time = time.time()

        for step in range(num_steps):
            # 获取当前信道状态
            current_channels = channel_data[step]
            state = np.reshape(current_channels, (STATE_DIM_A3C,))

            # A3C动作选择
            actions, logp, value = self.learner.select_action(state)
            decisions = np.array(actions)

            # 计算奖励
            processed_channels = self.channel_processor.process_channel(decisions, current_channels)
            reward = optimization(
                current_channels, processed_channels, decisions, self.runner
            )
            reward_value = reward 

            # 记录奖励
            self.reward_history.append(reward_value)

            # 创建轨迹并更新（单步更新）
            trajectory = [(state, actions, logp, reward_value, False, value)]
            loss = self.learner.update(trajectory)
            if loss > 0:
                self.loss_history.append(loss)

            # 进度显示
            if step % 1000 == 0:
                progress = (step / num_steps) * 100
                current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                    self.reward_history)
                print(f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}")

        total_time = time.time() - start_time
        print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

        return self.reward_history, self.loss_history, total_time


# PPO Networks
class PPOActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM_PPO, 300)
        self.fc2 = nn.Linear(300, 450)
        self.fc3 = nn.Linear(450, 300)
        self.fc4 = nn.Linear(300, 120)
        self.fc5 = nn.Linear(120, 100)
        self.out = nn.Linear(100, JOINT_ACTION_DIM)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        logits = self.out(x)
        logits = logits.view(-1, N, ACTION_PER_NODE)
        probs = F.softmax(logits, dim=-1)
        return probs


class PPOCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM_PPO, 120)
        self.fc2 = nn.Linear(120, 250)
        self.fc3 = nn.Linear(250, 400)
        self.fc4 = nn.Linear(400, 550)
        self.fc5 = nn.Linear(550, 700)
        self.out = nn.Linear(700, 1)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        v = self.out(x)
        return v.squeeze(-1)


class PPOLearner:
    """PPO算法学习器 - 近端策略优化"""

    def __init__(self):
        self.actor = PPOActorNet().to(device)
        self.critic = PPOCriticNet().to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=PPO_LR_ACTOR, weight_decay=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=PPO_LR_CRITIC)

        self.reset_buffer()
        self.reward_history = []
        self.loss_history = []

    def reset_buffer(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        """选择动作"""
        s = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
        probs = self.actor(s).squeeze(0)  # (N, A)
        actions = []
        logp_terms = []

        for i in range(N):
            dist = Categorical(probs[i])
            a = dist.sample()
            actions.append(int(a.item()))
            logp_terms.append(dist.log_prob(a))

        logp = torch.stack(logp_terms).sum()
        value = self.critic(s).squeeze(0)

        return actions, logp, value

    def store_transition(self, state, action, logprob, reward, done, value):
        """存储经验"""
        self.states.append(np.reshape(state, (STATE_DIM_PPO,)).copy())
        self.actions.append(list(action))
        self.logprobs.append(logprob.detach().cpu().item() if isinstance(logprob, torch.Tensor) else float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value))

    def compute_gae(self, last_value=0.0):
        """计算广义优势估计"""
        T = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]

            delta = self.rewards[t] + PPO_GAMMA * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + PPO_GAMMA * PPO_LAMBDA * nextnonterminal * lastgaelam
            adv[t] = lastgaelam

        returns = adv + np.array(self.values, dtype=np.float32)
        # 归一化优势函数
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(self):
        """PPO更新"""
        if len(self.states) < PPO_MINI_BATCH:
            return 0

        # 计算最后一个状态的价值用于bootstrap
        last_state = self.states[-1]
        with torch.no_grad():
            last_val = self.critic(torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)).item()

        # 计算GAE和returns
        advs, returns = self.compute_gae(last_value=last_val)

        # 准备tensor
        states = torch.tensor(np.stack(self.states), dtype=torch.float32, device=device)
        actions_arr = np.array(self.actions, dtype=np.int64)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        total_loss = 0

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, PPO_MINI_BATCH):
                mb_idx = indices[start:start + PPO_MINI_BATCH]
                mb_states = states[mb_idx]
                mb_size = mb_states.shape[0]

                # 计算当前策略的动作概率
                probs = self.actor(mb_states)
                logp_curr = torch.zeros((mb_size,), dtype=torch.float32, device=device)
                entropy_term = torch.zeros((mb_size,), dtype=torch.float32, device=device)

                for i_idx, idx in enumerate(mb_idx):
                    acts = actions_arr[idx]
                    sample_probs = probs[i_idx]
                    s_log = 0.0
                    s_ent = 0.0

                    for node in range(N):
                        p_node = sample_probs[node]
                        dist_node = Categorical(p_node)
                        a_node = int(acts[node])
                        s_log = s_log + dist_node.log_prob(torch.tensor(a_node, device=device))
                        s_ent = s_ent + dist_node.entropy()

                    logp_curr[i_idx] = s_log
                    entropy_term[i_idx] = s_ent

                # 优势函数和returns
                mb_advs = advs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_oldlog = old_logprobs[mb_idx]

                # PPO损失计算
                ratio = torch.exp(logp_curr - mb_oldlog)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean() - PPO_ENTROPY_COEF * entropy_term.mean()

                # Critic损失
                value_preds = self.critic(mb_states)
                critic_loss = F.mse_loss(value_preds, mb_returns)

                # 更新Actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer_actor.step()

                # 更新Critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), PPO_MAX_GRAD_NORM)
                self.optimizer_critic.step()

                total_loss += (actor_loss.item() + critic_loss.item())

        self.loss_history.append(total_loss)
        return total_loss

    def set_runner(self, runner):
        """设置运行器"""
        self.runner = runner


class PPOAlgorithmRunner(AlgorithmRunner):
    """PPO算法运行器"""

    def __init__(self):
        super().__init__(PPOLearner, "PPO")
        self.learner.set_runner(self.runner)

    def run(self, channel_data, num_steps=10000):
        start_time = time.time()
        update_interval = min(PPO_UPDATE_TIMESTEP, 256)  # 测试时使用较小的更新间隔

        for step in range(num_steps):
            # 获取当前信道状态
            current_channels = channel_data[step]
            state = np.reshape(current_channels, (STATE_DIM_PPO,))

            # PPO动作选择
            actions, logp, value = self.learner.select_action(state)
            decisions = np.array(actions)

            # 计算奖励
            processed_channels = self.channel_processor.process_channel(decisions, current_channels)
            reward = optimization(
                current_channels, processed_channels, decisions, self.runner
            )
            reward_value = reward 

            # 记录奖励
            self.reward_history.append(reward_value)

            # 存储经验
            self.learner.store_transition(state, actions, logp, reward_value, False, value)

            # 定期更新
            if len(self.learner.states) >= update_interval:
                loss = self.learner.update()
                if loss > 0:
                    self.loss_history.append(loss)
                self.learner.reset_buffer()

            # 进度显示
            if step % 1000 == 0:
                progress = (step / num_steps) * 100
                current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                    self.reward_history)
                print(f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}")

        # 最后更新剩余的经验
        if len(self.learner.states) > 0:
            loss = self.learner.update()
            if loss > 0:
                self.loss_history.append(loss)

        total_time = time.time() - start_time
        print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

        return self.reward_history, self.loss_history, total_time


# SAC Networks
class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM_SAC, SAC_HIDDEN_DIM)
        self.fc2 = nn.Linear(SAC_HIDDEN_DIM, SAC_HIDDEN_DIM)
        self.out = nn.Linear(SAC_HIDDEN_DIM, JOINT_ACTION_DIM)

    def forward(self, s):
        s = s.view(s.shape[0], -1)
        if s.shape[1] != STATE_DIM_SAC:
            raise ValueError(f"Actor input last-dim must be STATE_DIM_SAC={STATE_DIM_SAC}, got {s.shape}")
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        logits = logits.view(-1, N, ACTION_PER_NODE)
        probs = F.softmax(logits, dim=-1)
        return probs


class SACCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM_SAC + JOINT_ACTION_DIM, SAC_HIDDEN_DIM)
        self.fc2 = nn.Linear(SAC_HIDDEN_DIM, SAC_HIDDEN_DIM)
        self.out = nn.Linear(SAC_HIDDEN_DIM, 1)

    def forward(self, s, a_onehot):
        s = s.view(s.shape[0], -1)
        if s.shape[1] != STATE_DIM_SAC:
            raise ValueError(f"Critic state last-dim must be STATE_DIM_SAC={STATE_DIM_SAC}, got {s.shape}")
        x = torch.cat([s, a_onehot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        return q


class SACLearner:
    """SAC算法学习器 - 软Actor-Critic"""

    def __init__(self):
        self.actor = SACActor().to(device)
        self.critic1 = SACCritic().to(device)
        self.critic2 = SACCritic().to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.opt_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=SAC_LR)
        self.opt_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=SAC_LR)

        self.alpha = SAC_ALPHA
        self.gamma = SAC_GAMMA
        self.tau = SAC_TAU

        self.replay_buffer = deque(maxlen=20000)
        self.reward_history = []
        self.loss_history = []

    def select_action(self, state):
        """选择动作"""
        s = torch.tensor(np.reshape(state, (1, STATE_DIM_SAC)), dtype=torch.float32, device=device)
        probs = self.actor(s)
        actions = []

        for i in range(N):
            p = probs[0, i].detach().cpu()
            dist = Categorical(probs=p)
            a = dist.sample().item()
            actions.append(int(a))

        return actions

    def update(self, batch_size=64):
        """更新网络参数"""
        if len(self.replay_buffer) < batch_size:
            return 0

        batch = random.sample(self.replay_buffer, batch_size)
        s_list, a_list, r_list, s_next_list, done_list = zip(*batch)

        # 准备数据
        s_np = np.stack([np.reshape(x, (STATE_DIM_SAC,)) for x in s_list])
        s_next_np = np.stack([np.reshape(x, (STATE_DIM_SAC,)) for x in s_next_list])

        s = torch.tensor(s_np, dtype=torch.float32, device=device)
        s_next = torch.tensor(s_next_np, dtype=torch.float32, device=device)
        r = torch.tensor(np.array(r_list, dtype=np.float32)).unsqueeze(1).to(device)
        done = torch.tensor(np.array(done_list, dtype=np.float32)).unsqueeze(1).to(device)

        B = s.shape[0]

        # 构建动作one-hot编码
        a_onehot = torch.zeros((B, JOINT_ACTION_DIM), dtype=torch.float32, device=device)
        for i in range(B):
            act_i = a_list[i]
            for j in range(N):
                idx = j * ACTION_PER_NODE + int(act_i[j])
                a_onehot[i, idx] = 1.0

        total_loss = 0

        # --- 目标Q值计算 ---
        with torch.no_grad():
            probs_next = self.actor(s_next)
            probs_next_flat = probs_next.view(B, JOINT_ACTION_DIM)

            q1_next = self.target_critic1(s_next, probs_next_flat)
            q2_next = self.target_critic2(s_next, probs_next_flat)
            q_next = torch.min(q1_next, q2_next)

            target_q = r + self.gamma * (1.0 - done) * q_next

        # --- Critic损失和更新 ---
        q1 = self.critic1(s, a_onehot)
        q2 = self.critic2(s, a_onehot)

        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)

        self.opt_critic1.zero_grad()
        loss_q1.backward()
        self.opt_critic1.step()

        self.opt_critic2.zero_grad()
        loss_q2.backward()
        self.opt_critic2.step()

        total_loss += loss_q1.item() + loss_q2.item()

        # --- Actor更新 ---
        probs = self.actor(s)
        probs_flat = probs.view(B, JOINT_ACTION_DIM)

        log_probs = torch.log(probs + 1e-10)
        log_probs_flat = log_probs.view(B, JOINT_ACTION_DIM)

        q1_pi = self.critic1(s, probs_flat)
        q2_pi = self.critic2(s, probs_flat)
        q_pi = torch.min(q1_pi, q2_pi)

        ent = (probs_flat * log_probs_flat).sum(dim=1, keepdim=True)
        loss_actor = (self.alpha * ent - q_pi).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        total_loss += loss_actor.item()

        # --- 软更新目标网络 ---
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        self.loss_history.append(total_loss)
        return total_loss

    def set_runner(self, runner):
        """设置运行器"""
        self.runner = runner


class SACAlgorithmRunner(AlgorithmRunner):
    """SAC算法运行器"""

    def __init__(self):
        super().__init__(SACLearner, "SAC")
        self.learner.set_runner(self.runner)

    def run(self, channel_data, num_steps=10000):
        start_time = time.time()

        for step in range(num_steps):
            # 获取当前信道状态
            current_channels = channel_data[step]

            # SAC动作选择
            actions = self.learner.select_action(current_channels)
            decisions = np.array(actions)

            # 计算奖励
            processed_channels = self.channel_processor.process_channel(decisions, current_channels)
            reward = optimization(
                current_channels, processed_channels, decisions, self.runner
            )
            reward_value = reward 

            # 获取下一状态
            if step < num_steps - 1:
                next_state = channel_data[step + 1]
            else:
                next_state = current_channels

            # 存储经验
            self.learner.replay_buffer.append((
                np.reshape(current_channels, (STATE_DIM_SAC,)),
                list(actions),
                reward_value,
                np.reshape(next_state, (STATE_DIM_SAC,)),
                False
            ))

            # 记录奖励
            self.reward_history.append(reward_value)

            # 更新网络
            loss = self.learner.update(batch_size=64)
            if loss > 0:
                self.loss_history.append(loss)

            # 进度显示
            if step % 1000 == 0:
                progress = (step / num_steps) * 100
                current_avg = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(
                    self.reward_history)
                buffer_size = len(self.learner.replay_buffer)
                print(
                    f"{self.algorithm_name} Progress: {progress:.1f}%, Recent Avg Reward: {current_avg:.4f}, Buffer: {buffer_size}")

        total_time = time.time() - start_time
        print(f"{self.algorithm_name} completed in {total_time:.2f} seconds")

        return self.reward_history, self.loss_history, total_time

def main():
    """主函数 - 运行所有算法并比较结果"""
    # 加载数据
    try:
        # 请替换为你的实际数据路径
        channel_data = sio.loadmat('/Users/tongxun/PycharmProjects/MEC_PLS_WPT/N15/Gen/05.mat')['input_h']
        print(f"成功加载数据，数据形状: {channel_data.shape}")
    except Exception as e:
        print(f"无法加载数据文件: {e}")
        print("使用随机数据作为示例")
        # 生成随机数据作为示例
        channel_data = np.random.rand(10000, N * M)

    # 初始化算法运行器和结果比较器
    algorithms = [
        OriginalAlgorithmRunner(),
        PGAlgorithmRunner(),
        ACAlgorithmRunner(),
        DDPGAlgorithmRunner(),
        DROOAlgorithmRunner(),
        H2ACAlgorithmRunner(),  # 新增H2AC算法
        A3CAlgorithmRunner(),  # 新增A3C算法
        PPOAlgorithmRunner(),
        SACAlgorithmRunner()
    ]

    comparator = ResultComparator()

    # 运行所有算法
    for algorithm in algorithms:
        print(f"\n{'=' * 50}")
        print(f"开始运行 {algorithm.algorithm_name}")
        print(f"{'=' * 50}")

        # if hasattr(algorithm, 'k_history'):
        #     rewards, losses, training_time = algorithm.run(channel_data, num_steps=1000)
        #     comparator.add_result(algorithm.algorithm_name, rewards, losses, training_time, algorithm.k_history)
        # else:
        rewards, losses, training_time = algorithm.run(channel_data, num_steps=10000)
        comparator.add_result(algorithm.algorithm_name, rewards, losses, training_time)

        # # 保存结果到文件
        # filename = f"{algorithm.algorithm_name.replace(' ', '_').replace('(', '').replace(')', '')}_results.txt"
        # with open(filename, 'w') as f:
        #     for reward in rewards:
        #         f.write(f"{reward}\n")

        print(f"{algorithm.algorithm_name} 完成")

    # 生成对比结果
    print(f"\n{'=' * 60}")
    print("算法对比结果")
    print(f"{'=' * 60}")

    # 绘制对比图表
    comparator.plot_comparison(save_path='algorithm_comparison.png')
    print("对比图表已保存到 'algorithm_comparison.png'")

    # 输出详细统计信息
    print(f"\n{'=' * 50}")
    print("详细统计信息")
    print(f"{'=' * 50}")
    for algo_name, result in comparator.results.items():
        print(f"\n{algo_name}:")
        print(f"  平均奖励: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        print(f"  最佳奖励: {result['max_reward']:.4f}")
        print(f"  最差奖励: {result['min_reward']:.4f}")
        print(f"  最终奖励: {result['final_reward']:.4f}")
        print(f"  收敛步数: {result['convergence_step']}")
        if result['training_time']:
            print(f"  训练时间: {result['training_time']:.2f} 秒")


if __name__ == "__main__":
    main()