import gym
import random
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_episodes", type=int, default=50, help="number of epochs of training")
parser.add_argument("--eps_start", type=float, default=0.9, help="The probability that an agent behaves randomly at the start of learning")
parser.add_argument("--eps_end", type=float, default=0.05, help="The probability that an agent behaves randomly at the end of learning")
parser.add_argument("--eps_decay", type=int, default=200, help="A value that reduces the probability that an agent behaves randomly during learning")
parser.add_argument("--gamma", type=float, default=0.8, help="discount factor")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
opt = parser.parse_args()
print(opt)

class DQN_Agent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.optimizer = optim.Adam(self.model.parameters(), opt.lr)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((
            state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])
        ))

    def act(self, state):
        eps_threshold = opt.eps_end + (opt.eps_start - opt.eps_end) * math.exp(-1. * self.steps_done / opt.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(-1,1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def learn(self):
        if len(self.memory) < opt.batch_size:
            return
        batch = random.sample(self.memory, opt.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (opt.gamma * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

env = gym.make('CartPole-v0')
agent = DQN_Agent()
score_history = []

for e in range(1, opt.n_episodes+1):
    state = env.reset()
    steps = 0

    while True:
        env.render()
        state = torch.FloatTensor([state])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action.item())
        
        if done:
            reward = -1
            
        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1

        if done:
            print('Eposide:{0}, Score: {1}'.format(e, steps))
            score_history.append(steps)
            break