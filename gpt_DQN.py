import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 超参数
env_name = "CartPole-v1"
gamma = 0.99                # 折扣因子
lr = 0.001                  # 学习率
batch_size = 64             # 批次大小
epsilon_start = 1.0         # 初始 epsilon
epsilon_end = 0.01          # 最小 epsilon
epsilon_decay = 0.995       # epsilon 衰减
memory_size = 10000         # 经验回放大小
episodes = 500              # 训练轮数
target_update = 10          # 目标网络更新频率

# 初始化环境
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 初始化 Q 网络和目标网络
policy_net = QNetwork(state_size, action_size)
target_net = QNetwork(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = deque(maxlen=memory_size)

# epsilon-greedy 策略
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        if isinstance(state, tuple):
            state = state[0]
        else:
            state = state
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax().item()

# 经验回放的学习函数
def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    filtered_states = []
    for state in states:
        if isinstance(state, tuple):
            filtered_states.append(state[0])
        elif isinstance(state, np.ndarray):
            filtered_states.append(state)

    # 将过滤后的数组转换为张量
    states = torch.FloatTensor(np.array(filtered_states))
    # print("states:", states.shape)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # 计算 Q 值
    q_values = policy_net(states).gather(1, actions)

    # 计算目标 Q 值
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    # 损失函数
    loss = nn.MSELoss()(q_values, target_q_values)

    # 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 主训练循环
epsilon = epsilon_start
for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        # 选择动作
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储经验
        memory.append((state, action, [reward], next_state, done))
        state = next_state
        total_reward += reward

        # 优化模型
        optimize_model()

        if done:
            break

    # 更新 epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 更新目标网络
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()
