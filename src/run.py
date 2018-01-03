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

PATH = os.path.join(os.path.dirname(__file__), "./store")

def run(submit_key, gpu):
  env = Environment()
  agent = RenjuAgent(env.actions, epsilon=0.01, model_path=PATH, on_gpu=gpu)
  path = ""
  episode = 5
  if submit_key:
    print("make directory to submit result")
    path = os.path.join(os.path.dirname(__file__), "submit")
    episode = 100

  for episode_count, step_count, reward in env.play(agent, episode=episode, render=True, record_path=path):
    # TODO: render learning progress
    pass

  if submit_key:
    # TODO: Modify if want to upload result
    # gym.upload(path, api_key=submit_key)
    pass

def train(render, gpu):
  env = LearningEnvironment('Renju15x15-learning-noswap-v0')
  trainers = [RenjuTrainer(RenjuAgent(env.actions, epsilon=1, model_path=PATH, on_gpu=gpu)) for _ in range(2)]

  for episode_count, step_count, reward in env.play(trainers, episode=10**5, render=False, report_interval=25):
    # TODO: render learning progress
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