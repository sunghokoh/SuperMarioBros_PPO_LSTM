import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

import numpy as np
from statistics import mean

from Environment import MultipleEnv
from Model import ActorCriticNet

# settings
train_max_step = 3000000
eps_clip = 0.1
gamma = 0.99
lambd = 0.95
learning_rate = 3e-5

T_horizon = 32
K_epoch = 10
N_worker = 8

save_interval = 1000

world = 1
stage = 1

model_path = './Models/SuperMarioBros_PPO_LSTM_{}-{}.model'.format(world, stage)
history_path = './Train_Historys/train_history_PPO_LSTM_{}-{}'.format(world, stage)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(Net, optimizer, states, actions, rewards, dones, old_probs, final_state, start_h, start_c):
    states = torch.FloatTensor(states).to(device) # (T, N, 1, 84, 84)
    actions = torch.LongTensor(actions).view(-1, 1).to(device) # (T*N, 1)
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device) # (T*N, 1)
    dones = torch.FloatTensor(dones).to(device) # (T, N)
    old_probs = torch.FloatTensor(old_probs).view(-1, 1).to(device) # (T*N, 1)
    final_state = torch.FloatTensor(final_state).to(device)

    for _ in range(K_epoch):
        # Calculate Probs, values
        probs = []
        values = []
        h = start_h
        c = start_c

        for state, done in zip(states, dones):
            prob, value, (h, c)= Net(state, (h, c))
            probs.append(prob)
            values.append(value)
            for i, d in enumerate(done):
                if d.item() == 0:
                    h, c = reset_hidden(i, h, c)

        _, final_value, _ = Net(final_state, (h, c))
        next_values = values[1:]
        next_values.append(final_value)
            
        probs = torch.cat(probs)
        values = torch.cat(values)
        next_values = torch.cat(next_values)
    
        td_targets = rewards + gamma * next_values * dones.view(-1, 1) #(T*N, 1)
        deltas = td_targets - values # (T*N, 1)

        # calculate GAE
        deltas = deltas.view(T_horizon, N_worker, 1).cpu().detach().numpy() #(T, N, 1)
        masks = dones.view(T_horizon, N_worker, 1).cpu().numpy()
        advantages = []
        advantage = 0
        for delta, mask in zip(deltas[::-1], masks[::-1]):
            advantage = gamma * lambd * advantage * mask + delta
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device)

        probs_a = probs.gather(1, actions) #(T*N, 1)
        m = Categorical(probs)
        entropys = m.entropy()

        ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = F.smooth_l1_loss(values, td_targets.detach())
        entropy_loss = torch.mean(entropys)

        loss = actor_loss + critic_loss -0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c

def convert_actions_16(actions): # used when network outputs 16 actions
    action_list = [5,5,0,0,4,2,3,1,9,7,8,6,5,5,0,0]
    acts = []
    for action in actions:
        act = action_list[action]
        acts.append(act)
    return acts
    

def main():
    env = MultipleEnv(N_worker, world, stage, 'SIMPLE')
    Net = ActorCriticNet(7).to(device) # 7 : SIMPLE MOVEMENT, 10 : COMPLEX MOVEMENT
    #Net.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(Net.parameters(), learning_rate)

    scores = [0.0 for _ in range(N_worker)]
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()

    step = len(train_history) * 1000

    state = env.reset()
    h = torch.zeros(N_worker, 512).to(device)
    c = torch.zeros(N_worker, 512).to(device)

    print("Train Start")
    while step <= train_max_step:
        start_h = h
        start_c = c
        states, actions, rewards, dones, old_probs = list(), list(), list(), list(), list()
        for _ in range(T_horizon):
            prob, _, (next_h, next_c) = Net(torch.FloatTensor(state).to(device), (h, c))
            m = Categorical(prob)

            action = m.sample() # (N,)
            old_prob = prob.gather(1, action.unsqueeze(1)) # (N, 1)

            action = action.cpu().detach().numpy()
            old_prob = old_prob.cpu().detach().numpy()

            # action = convert_actions_16(action) # used when network outputs 16 actions
            next_state, reward, done = env.step(action) #(N, 1, 84, 84), (N,), (N,)

            # save transition
            states.append(state) # (T, N, 1, 84, 84)
            actions.append(action) # (T, N)
            rewards.append(reward/100.0) # (T, N)
            dones.append(1-done) # (T, N)
            old_probs.append(old_prob) # (T, N, 1)

            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                scores[i] += r

                if d==True:
                    score_history.append(scores[i])
                    scores[i] = 0.0
                    if len(score_history) > 100:
                        del score_history[0]
                    next_h, next_c = reset_hidden(i, next_h, next_c) # if done, reset hidden   
            
            state = next_state
            h = next_h.detach()
            c = next_c.detach()

            step += 1

            if step % save_interval == 0:
                train_history.append(mean(score_history))
                torch.save(Net.state_dict(), model_path)
                np.save(history_path, np.array(train_history))
                print("step : {}, world {}-{}, Average score of last 100 episode : {:.1f}".format(step, world, stage, mean(score_history)))
        
        train(Net, optimizer, states, actions, rewards, dones, old_probs, state, start_h, start_c)
        
        
    torch.save(Net.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main()