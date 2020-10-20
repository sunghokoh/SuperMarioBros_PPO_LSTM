from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym import Wrapper

import numpy as np
import random
import cv2
import multiprocessing as mp

class CustomFrameProcess(Wrapper):
    def __init__(self, env, size=84, skip=4):
        super(CustomFrameProcess, self).__init__(env)
        self.skip = skip
        self.size = size
        self.running = False

    def ProcessFrame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.size, self.size)) / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        state = self.ProcessFrame(state)
        return state, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.ProcessFrame(state)
        return state

def CreateEnv(world, stage, ACTION):
    env = gym_super_mario_bros.make('SuperMarioBros-{}-{}-v0'.format(world, stage))
    if ACTION == 'SIMPLE':
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if ACTION == 'COMPLEX':
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustomFrameProcess(env)
    return env

class MultipleEnv:
    def __init__(self, N, world, stage, ACTION):
        self.world = world
        self.stage = stage
        self.envs = [CreateEnv(self.world, self.stage, ACTION) for _ in range(N)]

    def reset(self):
        obs = []
        for env in self.envs:
            ob = env.reset()
            obs.append(ob)
        return np.stack(obs)
    
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            if done:
                ob = env.reset()
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.stack(rewards), np.stack(dones)
    
    def render(self):
        for env in self.envs:
            env.render()