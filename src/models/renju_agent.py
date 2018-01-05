# -*- coding:utf-8 -*-

"""Agent module for Renju."""

# Imports
import os
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from typing import List

from src.models.agent import Agent

# Size of board 
SIZE = 15 # Use 'Renju' size

class RenjuQ(chainer.Chain):
  '''
  Q-learning model of CNN for Renju.
  '''

  # NN model
  L1_OUT = 8
  L2_OUT = 16
  L3_FLATTEN = 400
  L3_OUT = 512

  def __init__(self, num_history, num_action, on_gpu=False):
    self._num_history = num_history
    self._num_action = num_action
    self._on_gpu = on_gpu
    # Init Layers
    super(RenjuQ, self).__init__()
    with self.init_scope():
      initializer = chainer.initializers.HeNormal()
      self._l1 = L.Convolution2D(num_history, self.L1_OUT, ksize=9, stride=3, pad=4,
       initialW=initializer, nobias=False)
      self._l2 = L.Convolution2D(self.L1_OUT, self.L2_OUT, ksize=3, stride=1, pad=1,
       initialW=initializer, nobias=False)
      self._l3 = L.Linear(self.L3_FLATTEN, self.L3_OUT,
       initialW=initializer)
      self._out = L.Linear(self.L3_OUT, num_action,
      #  initialW=initializer)
       initialW=np.zeros((num_action, self.L3_OUT), dtype=np.float32))

    if on_gpu:
      self.to_gpu()

  def __call__(self, state: np.ndarray):
    _state = self.arr_to_gpu(state)
    s = chainer.Variable(_state)
    h1 = F.relu(self._l1(s))
    h2 = F.relu(self._l2(h1))
    h3 = F.relu(self._l3(h2))
    q_value = self._out(h3)
    return q_value

  def arr_to_gpu(self, arr):
    return chainer.cuda.to_gpu(arr) if self._on_gpu else arr

  def get_num_history(self) -> int:
    return self._num_history

  def get_num_action(self) -> int:
    return self._num_action

  def get_on_gpu(self) -> bool:
    return self._on_gpu

class RenjuAgent(Agent):
  '''
  Q-learning agent for Renju
  '''
  
  def __init__(self, actions: List, epsilon: float = 1.0, num_history: int = 4,
   on_gpu: bool = False, model_path: str = "", load_if_exist: bool = True):
    '''
    Initialize agent with learning parameters.
    @param actions: List of action candidates
    @param epsilon: epsilon rate for choosing random action in e-greedy
    @param num_history: number of state history used in each learning
    '''
    self._actions = actions
    self._epsilon = epsilon
    self._q = RenjuQ(num_history, len(actions), on_gpu) # TODO: !NO AVAILABLE GPU CALCULATION ON MAC book Air!
    # Preserve each state of repetitions
    self._state = []
    # Observations of current state and previous state, for FixedTarget
    self._observations = [
      np.zeros((SIZE, SIZE), np.float32),
      np.zeros((SIZE, SIZE), np.float32)
    ]
    self._last_action = 0
    self._model_path = model_path if model_path else os.path.join(os.path.dirname(__file__), "./../store")
    if not os.path.exists(self._model_path):
        print("make directory to store model at {0}".format(self._model_path))
        os.mkdir(self._model_path)
    else:
        models = self.get_model_files()
        if load_if_exist and len(models) > 0:
            print("load model file {0}.".format(models[-1]))
            chainer.serializers.load_npz(os.path.join(self._model_path, models[-1]), self._q)  # use latest model

  def _update_state(self, observation) -> any:
    # Why take maximum? -> Because original version takes images and compensate for clipped area.
    # So we do not need it for renju
    # formatted = self._format(obervation)
    # formatted = observation
    # state = np.maximum(formatted, self._observations[0])    
    # pop queue if the size is over number of history
    state = observation.astype(np.float32)
    self._state.append(state)
    if len(self._state) > self._q.get_num_history():
      self._state.pop(0)
    return state

  def reset(self, observation, valid_actions) -> int:
    self._state = []
    self._observations = [
      np.zeros((SIZE, SIZE), np.float32),
      np.zeros((SIZE, SIZE), np.float32)
    ]
    self._last_action = 0

  def act(self, observation, reward, valid_actions) -> int:
    o = self._update_state(observation)
    s = self.get_state()
    # Feedforward with batch size 1
    qv = self._q(np.array([s]))
    
    # e-greedy
    if np.random.rand() < self._epsilon:
      action = np.random.choice(valid_actions, replace=False)
    else:
      action = valid_actions[np.argmax(qv.data[-1][valid_actions])]

    self._observations[-1] = self._observations[0].copy()
    self._observations[0] = o
    self._last_action = action

    return action

  def get_state(self) -> None:
    state = []
    for i in range(self._q.get_num_history()):
      if i < len(self._state):
        state.append(self._state[i])
      else:
        # Zero padding
        state.append(np.zeros((SIZE, SIZE), dtype=np.float32))
    return np.array(state)

  def save(self, index: int = 0) -> None:
    fname = "renju.model" if index == 0 else "renju_{0}.model".format(index)
    path = os.path.join(self._model_path, fname)
    chainer.serializers.save_npz(path, self._q)

  def get_model_files(self) -> List:
    files = os.listdir(self._model_path)
    model_files = []
    for f in files:
      if f.startswith("renju") and f.endswith(".model"):
        model_files.append(f)

    model_files.sort()
    return model_files

  def get_num_history(self) -> int:
    return self._q.get_num_history()

  def get_num_action(self) -> int:
    return self._q.get_num_action()

  def get_on_gpu(self) -> bool:
    return self._q.get_on_gpu()
  
  def get_q(self) -> RenjuQ:
    return self._q

  def get_epsilon(self) -> float:
    return self._epsilon

  def set_epsilon(self, epsilon: float) -> None:
    self._epsilon = epsilon

  def get_last_action(self) -> int:
    return self._last_action