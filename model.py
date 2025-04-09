import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# 定義經驗元組
Experience = namedtuple('Experience', ('state', 'action_probs', 'reward', 'next_state', 'action'))

class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_probs, reward, next_state, action):
        """添加經驗到緩衝區"""
        self.buffer.append(Experience(state, action_probs, reward, next_state, action))
    
    def sample(self, batch_size):
        """隨機抽樣經驗"""
        experiences = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*experiences))
        return batch
    
    def __len__(self):
        return len(self.buffer)

class TripleTownNet(nn.Module):
    def __init__(self, device, board_size=6, num_piece_types=22):
        super(TripleTownNet, self).__init__()
        self.device = device
        
        # 輸入特徵處理
        self.board_size = board_size
        self.input_channels = board_size * board_size + 1
        self.embedding_size = 8
        
        # 使用全連接層替代卷積層
        self.fc1 = nn.Linear(self.input_channels * self.embedding_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        
        # 策略頭 - 預測每個位置的行動概率
        self.policy_fc1 = nn.Linear(256, 128)
        self.policy_fc2 = nn.Linear(128, board_size * board_size)
        
        # 價值頭 - 預測狀態的價值
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.to(self.device)

    
    def forward(self, state):

        x = state
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 256)
        
        # 策略頭
        policy_logits = F.relu(self.policy_fc1(x))
        policy_logits = self.policy_fc2(policy_logits)
        
        # 價值頭
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        return policy_logits, value

class TripleTownPredict(nn.Module):
    def __init__(self, device, action_space = 36, board_size=6, num_piece_types=22):
        super(TripleTownPredict, self).__init__()
        
        # 輸入特徵處理
        self.action_space = action_space
        self.device = device
        self.board_size = board_size
        self.input_channels = board_size * board_size + 1
        self.embedding_size = 8
        self.embedding = nn.Embedding(num_piece_types, self.embedding_size).to(self.device)
        
        # 使用全連接層替代卷積層
        self.fc1 = nn.Linear(self.input_channels * self.embedding_size + self.action_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, self.input_channels * self.embedding_size)

    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(state, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            x = state

        embedding_x = self.embedding(x)
        x = embedding_x.view(x.size(0), -1)
        action_onehot = F.one_hot(action, num_classes=self.action_space)
        x = torch.cat((x, action_onehot), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)
        return x , embedding_x
        
