# -*- coding:utf-8 -*-

"""Environment module for renju gym."""

# Imports
import os
import sys
import gym
import gym_renju
import numpy as np
from typing import Generator, List
from src.models.agent import Agent
from src.models import renju_agent

class LearningEnvironment():

  def __init__(self, env_name="Renju15x15-learning-noswap-v0") -> None:
    '''Initialize environment for renju gym.'''
    self.env = gym.make(env_name)
    self.actions = list(range(self.env.action_space.n))

  def play(self, agents: List[Agent], episode: int = 5, render: bool = True, report_interval: int = -1,
   record_path: str = "") -> Generator:
    '''
    Play on gym environment.
    @param agents: List of agents for WHITE and BLACK
    @param episode: Number of games (episodes) to play
    @param render: If render result to prompt or not
    @param report_interval: Length of interval to record result. Not record when is lower than 1
    @param record_path: File path to record result. Set default path if not specified.
    '''
    assert len(agents) == 2

    scores = []
    if record_path:
      self.env.monitor.start(record_path)

    for i in range(episode):
      observation = self.env.reset()
      observation = np.reshape(observation, (renju_agent.SIZE, renju_agent.SIZE))
      done = False
      reward = 0.0
      step_count = 0
      score = 0.0
      continue_game = True

      for agent in agents:
        agent.reset(observation, self.env.action_space.valid_spaces)

      agent_index = 0
      while continue_game:
        if render:
          self.env.render()

        # Select action
        agent = agents[agent_index]
        action = agent.act(observation, reward, self.env.action_space.valid_spaces)

        # Proceed state
        observation, reward, done, info = self.env.step(action)
        observation = np.reshape(observation, (renju_agent.SIZE, renju_agent.SIZE))

        # Finish game
        if done:
          agent.end(observation, reward, self.env.action_space.valid_spaces)

        yield i, step_count, reward

        continue_game = not done
        score += reward
        step_count += 1

      scores.append(score)

      if report_interval > 0 and np.mod(i, report_interval) == 0:
        print("average score is {0}.".format(sum(scores) / len(scores)))
        report = agent.report(i)
        if report:
          print(report)
        scores = []

    if record_path:
      self.env.monitor.close()