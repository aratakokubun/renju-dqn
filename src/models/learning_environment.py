# -*- coding:utf-8 -*-

"""Environment module for renju gym."""

# Imports
import os
import sys
import gym
import gym_renju
import numpy as np
from typing import Generator, List
from datetime import datetime as dt

from src.models.agent import Agent
from src.models import renju_agent

FILE_FORMAT = './store/logs/renju_cnn_{0}_{1}.log'

class LearningEnvironment():

  def __init__(self, env_name="Renju15x15-learning-noswap-v0") -> None:
    '''Initialize environment for renju gym.'''
    self.env = gym.make(env_name)
    self.actions = list(range(self.env.action_space.n))
    datetime_str = dt.now().strftime('%Y%m%d%H%M%S')
    self.file_paths = [
      FILE_FORMAT.format(datetime_str, 'BLACK'),
      FILE_FORMAT.format(datetime_str, 'WHITE'),
    ]

  def save(self, report: str, file_path: str) -> None:
    with open(file_path, 'a') as myfile:
      myfile.write(report)
      myfile.write('\n')

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

    total_scores = [[], []]
    if record_path:
      self.env.monitor.start(record_path)

    for i in range(episode):
      # Reshape observation to 2d array for Convolutional Neural Net
      observation = np.reshape(self.env.reset(), (renju_agent.SIZE, renju_agent.SIZE))
      done = False
      rewards = [0.0, 0.0]
      step_count = 0
      scores = [0.0, 0.0]
      continue_game = True

      for agent in agents:
        agent.reset(observation, self.env.action_space.valid_spaces)

      agent_index = 0
      while continue_game:
        if render:
          self.env.render()

        # Select action
        agent = agents[agent_index]
        action = agent.act(observation, rewards[agent_index], self.env.action_space.valid_spaces)

        # Proceed state
        observation, reward, done, info = self.env.step(action)
        observation = np.reshape(observation, (renju_agent.SIZE, renju_agent.SIZE))

        # Finish game
        if done:
          agent.end(observation, reward, self.env.action_space.valid_spaces)

        yield i, step_count, reward

        continue_game = not done
        rewards[agent_index] = reward
        scores[agent_index] += reward
        step_count += 1
        agent_index = (agent_index + 1) % 2

      # Get the last reward of the next player
      last_rewards = self.env.get_rewards()
      scores[agent_index] += last_rewards[agent_index]

      total_scores[0].append(scores[0])
      total_scores[1].append(scores[1])

      if report_interval > 0 and np.mod(i, report_interval) == 0:
        ave_scores = [sum(total)/len(total) for total in total_scores]
        print("average score is [<play first>:{0}, <draw first>{1}].".format(ave_scores[0], ave_scores[1]))
        for agent_index in range(2):
          report = agent.report(i)
          if report:
            print(report)
            self.save(report + ", score={0}".format(ave_scores[agent_index]), self.file_paths[agent_index])
        total_scores = [[], []]

    if record_path:
      self.env.monitor.close()