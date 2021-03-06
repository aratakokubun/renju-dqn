# -*- coding:utf-8 -*-

# Imports
import gym
# Change import on installed on pip
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import gym_renju
env = gym.make('Renju15x15-v0') # default 'beginner' level opponent policy

env.reset()
env.render()
# env.step(15) # place a single stone, black color first

# play a game
env.reset()
for _ in range(225):
	action = env.action_space.sample() # sample without replacement
	observation, reward, done, info = env.step(action)
	env.render()
	if done:
		print ("Game is Over")
		break
