# -*- coding:utf-8 -*-

"""Trainer module for Renju DQN agent."""

# Imports
import numpy as np
from chainer import Variable
from chainer import optimizers
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from typing import List

from src.models.agent import Agent
import src.models.renju_agent as RenjuAgent
from src.models.renju_agent import RenjuQ

class RenjuTrainer(Agent):

  def __init__(self, agent: Agent, memory_size: int = 10**4, replay_size: int = 32,
   gamma: float = 0.99, initial_exploration: int = 10**4, target_update_freq: int = 10**4,
   learning_rate: float = 0.00025, epsilon_decay: float = 1e-6,
   minimum_epsilon: float = 0.1) -> None:
    '''
      Trainer for Renju Agent class.
      @param memory_size:
      @param replay_size:
      @param gamma: decay rate of time
      @param initial_exploration:
      @param target_update_freq: frequency rate for update weights
      @param learning_rate: learning rate for TD error
      @param epsilon_decay: decay rate of epsilon in epsilon-greedy
      @param minimu_epsilon: minimum epsilon in epsilon-greedy after decay
    '''
    self._agent = agent
    self._target = RenjuQ(self._agent.get_num_history(), self._agent.get_num_action(),
     on_gpu=self._agent.get_on_gpu())

    self._memory_size = memory_size
    self._replay_size = replay_size
    self._gamma = gamma
    self._initial_exploration = initial_exploration
    self._target_update_freq = target_update_freq
    self._learning_rate = learning_rate
    self._epsilon_decay = epsilon_decay
    self._minimum_epsilon = minimum_epsilon
    self._step = 0

    # Prepare memory for replay
    num_history = self._agent.get_num_history()
    size = RenjuAgent.SIZE
    self._memory = [
      np.zeros((memory_size, num_history, size, size), dtype=np.float32),
      np.zeros(memory_size, dtype=np.uint8),
      np.zeros((memory_size, 1), dtype=np.float32),
      np.zeros((memory_size, num_history, size, size), dtype=np.float32),
      np.zeros((memory_size, 1), dtype=np.bool)
    ]
    self._memory_text = [
        "state", "action", "reward", "next_state", "episode_end"
    ]

    # Prepare optimize
    self._optimizer = optimizers.RMSpropGraves(lr=learning_rate, alpha=0.95, momentum=0.95, eps=0.01)
    self._optimizer.setup(self._agent.get_q())
    self._loss = 0
    self._qv = 0

  def calc_loss(self, states: List, actions: List, rewards: List, next_states: List, episode_ends: List):
    q = self._agent.get_q()
    # Feedforward and get current value
    qv = q(states)
    # Feedforward and get target value(learn to this target)
    q_t = self._target(next_states) # Q(s', *)
    # Calculate maximum E of RenjuQ
    max_q_prime = np.array(list(map(np.max, q_t.data)), dtype = np.float32) # max_a Q(s', a)

    target = cuda.to_cpu(qv.data.copy())
    for i in range(self._replay_size):
      # Calculate partial derivative
      # d(Loss)/d(Theta) = R(s,a) + gamma * E[Q(s',a')]
      # sign means Clipping for rewards
      # Note: reason for [i][0] is using 2d array for episode_end memory
      if episode_ends[i][0] is True:
        _r = np.sign(rewards[i])
      else:
        _r = np.sign(rewards[i]) + self._gamma * max_q_prime[i]

      target[i, actions[i]] = _r

    # Calculate td loss
    td = Variable(self._target.arr_to_gpu(target)) - qv
    # Add value to avoid zero division
    td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
    td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

    zeros = Variable(self._target.arr_to_gpu(np.zeros((self._replay_size, self._target.get_num_action()), dtype=np.float32)))
    # zeros means just take td_clip as erros 
    loss = F.mean_squared_error(td_clip, zeros)
    self._loss = loss.data
    self._qv = np.max(qv.data)
    return loss

  def start(self, observation, valid_actions):
    return self._agent.start(observation, valid_actions)

  def act(self, observation, reward, valid_actions) -> List:
    # For initial steps, prior to explore unknown actions and decay the probability gradually of this exploration
    if self._initial_exploration <= self._step:
      self._agent.set_epsilon(self._agent.get_epsilon() - 1.0/10**6)
      if self._agent.get_epsilon() < self._minimum_epsilon:
        self._agent.set_epsilon(self._minimum_epsilon)

    return self._train(observation, reward, valid_actions, episode_end=False)

  def end(self, observation, reward, valid_actions):
    self._train(observation, reward, valid_actions, episode_end=True)

  def _train(self, observation, reward, valid_actions, episode_end: bool) -> List:
    action = 0
    last_state = self._agent.get_state()
    last_action = self._agent.get_last_action()
    if episode_end:
      # Gym ignore it. And the action MUST be same for end to calculate Q table.
      self.memorize(last_state, last_action, reward, last_state, True)
    else:
      action = self._agent.act(observation, reward, valid_actions)
      result_state = self._agent.get_state()
      self.memorize(last_state, last_action, reward, result_state, False)

    if self._initial_exploration <= self._step:
      self.experience_replay()

      # Fixed Target Q-Network
      # Use same target(teacher data) for freq times, and update on each freq.
      if np.mod(self._step, self._target_update_freq) == 0:
        self._target.copyparams(self._agent.get_q())

    self._step += 1
    return action

  def memorize(self, state: List, action: int, reward, next_state: List, episode_end: bool):
    _index = np.mod(self._step, self._memory_size)
    self._memory[0][_index] = state
    self._memory[1][_index] = action
    self._memory[2][_index] = reward
    if not episode_end:
      self._memory[3][_index] = next_state
    self._memory[4][_index] = episode_end

  def experience_replay(self) -> None:
    # Experience replay: Randomize data to eliminate co-relation between time sequential data
    indices = []
    if self._step < self._memory_size:
      indices = np.random.randint(0, self._step, (self._replay_size))
    else:
      indices = np.random.randint(0, self._memory_size, (self._replay_size))

    states = []
    actions = []
    rewards = []
    next_states = []
    episode_ends = []
    for i in indices:
      states.append(self._memory[0][i])
      actions.append(self._memory[1][i])
      rewards.append(self._memory[2][i])
      next_states.append(self._memory[3][i])
      episode_ends.append(self._memory[4][i])

    to_np = lambda arr: np.array(arr)
    self._optimizer.target.cleargrads()
    loss = self.calc_loss(to_np(states), to_np(actions), to_np(rewards), to_np(next_states), to_np(episode_ends))
    loss.backward()

    # Update RMSpropsGrave
    self._optimizer.update()

  def report(self, episode) -> str:
    s = "{0}: loss={1}, q value={2}, epsilon={3}".format(self._step, self._loss, self._qv, self._agent.get_epsilon())
    self._agent.save(episode)
    return s
