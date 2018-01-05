# -*- coding:utf-8 -*-

"""Runner for renju."""

# Imports
import os
import sys
import argparse
import gym
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.models.environment import Environment
from src.models.learning_environment import LearningEnvironment
from src.models.renju_agent import RenjuAgent
from src.models.renju_trainer import RenjuTrainer

PATH = os.path.join(os.path.dirname(__file__), "../store")
STORE_PATHS = [
  os.path.join(os.path.dirname(__file__), "../store/white"),
  os.path.join(os.path.dirname(__file__), "../store/black"),
]

def run(submit_key, gpu):
  env = Environment()
  agent = RenjuAgent(env.actions, epsilon=0.01, model_path=STORE_PATHS[0], on_gpu=gpu)
  path = ""
  episode = 5
  if submit_key:
    print("make directory to submit result")
    path = os.path.join(os.path.dirname(__file__), "submit")
    episode = 100

  for ret in env.play(agent, episode=episode, render=True, record_path=path):
    pass

  if submit_key:
    # If want to upload result, call below
    # gym.upload(path, api_key=submit_key)
    pass

def train(render, gpu):
  # Declare environment for learnig (input vs input) and use same ends(play first or draw first) in all games.
  env = LearningEnvironment('Renju15x15-learning-noswap-v0')
  # Trainers for [0]: <player first> and [1]: <draw first>
  trainers = [RenjuTrainer(RenjuAgent(env.actions, epsilon=1, model_path=STORE_PATHS[i], on_gpu=gpu)) for i in range(2)]
  # Player games loop for <episode> times
  for ret in env.play(trainers, episode=10**5, render=False, report_interval=25):
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Renju DQN")
  parser.add_argument("--render", action="store_const", const=True, default=False, help="render or not")
  parser.add_argument("--submit", type=str, default="", help="api key to submit data")
  parser.add_argument("--train", action="store_const", const=True, default=False, help="user gpu or not")
  parser.add_argument("--gpu", action="store_const", const=True, default=False, help="user gpu or not")
  args = parser.parse_args()

  if args.train:
    train(args.render, args.gpu)
  else:
    run(args.submit, args.gpu)