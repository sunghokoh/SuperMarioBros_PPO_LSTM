import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

import numpy as np
from statistics import mean

from Environment import MultipleEnv
from Model import ActorCriticNet_MA

# settings
train_max_step = 3000000
eps_clip = 0.1
gamma = 0.99
lambd = 0.95
learning_rate = 1e-5

T_horizon = 32
K_epoch = 10
N_worker = 8

save_interval = 1000

world = 1
stage = 1

model_path = './Models/SuperMarioBros_MA_{}-{}_COMPLEX_NOENTROPY.model'.format(world, stage)
history_path = './Train_historys/train_history_MA_{}-{}_COMPLEX_NOENTROPY'.format(world, stage)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(Net, optimizer, states, actions, rewards, dones, old_action_probs, final_state, start_h, start_c):
    states = torch.FloatTensor(states).to(device) # (T, N, 1, 84, 84)
    actions = torch.LongTensor(actions).view(-1, 4).to(device) # (T*N, 4)
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device) # (T*N, 1)
    dones = torch.FloatTensor(dones).to(device) # (T, N)
    old_action_probs = torch.FloatTensor(old_action_probs).view(-1, 1).to(device) # (T*N, 1)
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
            
        probs = torch.cat(probs) # (T*N, 4)
        values = torch.cat(values) # (T*N, 1)
        next_values = torch.cat(next_values) # (T*N, 1)
    
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

        new_action_probs = torch.prod(actions*probs + (1-actions)*(1-probs), dim=1, keepdim=True) # (T*N, 1)

        ratio = torch.exp(torch.log(new_action_probs) - torch.log(old_action_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
        
        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = F.smooth_l1_loss(values, td_targets.detach())

        loss = actor_loss + critic_loss

        # when using entropy loss
        """
        entropys = torch.sum(-probs*torch.log(probs) + -(1-probs)*torch.log(1-probs), dim=1)
        entropy_loss = torch.mean(entropys)
        loss = loss -0.01 * entropy_loss
        """

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c

def convert_action(action):
    # Priority: right = left > A > B
    # action : [right, left, A, B]
    if action[0]:
        if action[1]:
            if action[2]:
                return 5 # A
            else:
                return 0 # NOOP
        else:
            if action[2]: 
                if action[3]: 
                    return 4 # right A B
                else:
                    return 2 # right A
            else:
                if action[3]:
                    return 3 # right B
                else:
                    return 1 # right
    else:
        if action[1]:
            if action[2]:
                if action[3]:
                    return 9 # left A B
                else:
                    return 7 # left A
            else:
                if action[3]:
                    return 8 # left B
                else:
                    return 6 # left
        else:
            if action[2]:
                return 5 # A
            else:
                return 0 # NOOP

def convert_actions(actions):
    acts = []
    for action in actions:
        act = convert_action(action)
        acts.append(act)
    return acts
    
def main():
    env = MultipleEnv(N_worker, world, stage, 'COMPLEX')
    Net = ActorCriticNet_MA().to(device)
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
        states, actions, rewards, dones, old_action_probs = list(), list(), list(), list(), list()
        for _ in range(T_horizon):
            prob, _, (next_h, next_c) = Net(torch.FloatTensor(state).to(device), (h, c))

            action = torch.bernoulli(prob)
            old_action_prob = torch.prod(action*prob + (1-action)*(1-prob), dim=1, keepdim=True) # (N, 1)

            action = action.cpu().detach().numpy()
            old_action_prob = old_action_prob.cpu().detach().numpy()

            next_state, reward, done = env.step(convert_actions(action)) #(N, 1, 84, 84), (N,), (N,)

            # save transition
            states.append(state) # (T, N, 1, 84, 84)
            actions.append(action) # (T, N, 4)
            rewards.append(reward/100.0) # (T, N)
            dones.append(1-done) # (T, N)
            old_action_probs.append(old_action_prob) # (T, N, 1)

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
        
        train(Net, optimizer, states, actions, rewards, dones, old_action_probs, state, start_h, start_c)
        
        
    torch.save(Net.state_dict(), model_path)
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))

if __name__ == "__main__":
    main()