import numpy
import time
import random

import torch

from Environment import CreateEnv
from Model import ActorCriticNet

world = 1
stage = 1

model_path = './Models/SuperMarioBros_PPO_LSTM_1-1.model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_action_16(actions): # used when networks outputs 16 action
    action_list = [5,5,0,0,4,2,3,1,9,7,8,6,5,5,0,0]
    return action_list[actions]

def Test():
    env = CreateEnv(world, stage, 'SIMPLE')
    Net = ActorCriticNet(7).to(device) # 7 : SIMPLE MOVEMENT, 10 : COMPLEX MOVEMENT
    Net.load_state_dict(torch.load(model_path))

    score = 0
    state = env.reset()
    h = torch.zeros(1, 512).to(device)
    c = torch.zeros(1, 512).to(device)
    done = False
        
    while not done:
        prob, _, (next_h, next_c) = Net(torch.FloatTensor([state]).to(device), (h, c))
        action = torch.argmax(prob).item()
        #action = convert_action_16(action) # used when networks outputs 16 action

        next_state, reward, done, info = env.step(action)
        env.render()

        score += reward
            
        state = next_state
        h = next_h.detach()
        c = next_c.detach()
        time.sleep(0.03)
        
    time.sleep(2)
        

    print('score : {}'.format(score))

if __name__ == "__main__":
    Test()
