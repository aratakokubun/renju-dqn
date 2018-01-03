# -*- coding:utf-8 -*-

"""Reinforcement agent."""

# Imports


class Agent(object):

  def __init__(self, actions):
    self.actions = actions

  def reset(self, observation, valid_actions):
    # Reset the state of agent
    return 0

  def act(self, observation, reward, valid_actions):
    # Default action
    return 0

  def end(self, observation, reward, valid_actions):
    pass

  def report(self, episode):
  # Report out the current state
    return ""
