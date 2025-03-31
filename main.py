from triple_town_simulate import TripleTownSim
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# 超參數
GAMMA = 0.99  # 折扣因子
LR = 0.001  # 學習率
EPSILON = 0.1  # 探索機率
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10  # 更新目標網絡的頻率

# 簡單的 DQN 網絡
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 簡單的 DQN 代理
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.policy_net = DQN(state_dim, action_dim).cuda()
        self.target_net = DQN(state_dim, action_dim).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

    def select_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, self.action_dim - 1)  # 隨機探索
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0)
                return torch.argmax(self.policy_net(state)).item()  # 選擇最優動作

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state), dtype=torch.float32).cuda()
        action = torch.tensor(action, dtype=torch.long).cuda().unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).cuda()
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).cuda()
        done = torch.tensor(done, dtype=torch.float32).cuda()

        q_values = self.policy_net(state).gather(1, action).squeeze()
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        target_q_values = reward + (1 - done) * GAMMA * next_q_values

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def load_model(self):
        if not os.path.exists("model.pth"):
            return
        self.policy_net.load_state_dict(torch.load("model.pth"))
        self.target_net.load_state_dict(torch.load("model.pth"))

# 訓練 DQN
env = TripleTownSim()

# load model

hand_item = 8
stock_item = 9
slot_item = 21 * 35
agent = DQNAgent(state_dim=37, action_dim=6*6)
agent.load_model()

for episode in range(1000):
    state = env.reset()
    total_reward = 0

    for t in range(3000):  # 限制最大回合數，防止無限循環
        action = agent.select_action(state)
        next_state = env.next_state_simulate(state, action)

        if np.array_equal(next_state, state):
            reward = -1
        else:
            reward = env.game_score

        done = env.is_end(next_state)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        if done:
            break

    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()

    print(f"Episode {episode}, Total Reward: {env.game_score}")

# save the model
torch.save(agent.policy_net.state_dict(), "model.pth")
print("訓練完成！")
