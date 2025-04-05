import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# 定義經驗元組
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加經驗到緩衝區"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """隨機抽樣經驗"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TripleTownDQN(nn.Module):
    def __init__(self):
        super(TripleTownDQN, self).__init__()
        # 基本參數
        self.board_size = 6
        self.num_item_types = 22
        self.embedding_dim = 8  # 增加嵌入維度
        
        # 物品嵌入層
        self.item_embedding = nn.Embedding(self.num_item_types, self.embedding_dim)
        
        # 卷積網絡 - 更深層次且有批標準化
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # 當前物品處理
        self.item_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # 價值和優勢分離 (Dueling DQN架構)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128 * self.board_size * self.board_size + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 36)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128 * self.board_size * self.board_size + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, state):
        batch_size = state.size(0)
        
        # 分離當前物品和遊戲板
        current_item = state[:, 0].long()
        board = state[:, 1:].reshape(batch_size, self.board_size, self.board_size).long()
        
        # 處理當前物品
        item_embedded = self.item_embedding(current_item)
        item_features = self.item_fc(item_embedded)
        
        # 處理遊戲板
        board_embedded = self.item_embedding(board)
        board_embedded = board_embedded.permute(0, 3, 1, 2)
        board_features = self.conv_layers(board_embedded)
        board_features = board_features.reshape(batch_size, -1)
        
        # 合併特徵
        combined_features = torch.cat([board_features, item_features], dim=1)
        
        # Dueling架構
        advantage = self.advantage_stream(combined_features)
        value = self.value_stream(combined_features)
        
        # Q值 = 價值 + (優勢 - 平均優勢)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class TripleTownNet(nn.Module):
    def __init__(self, board_size=6, num_piece_types=22):
        super(TripleTownNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 輸入特徵處理
        self.board_size = board_size
        self.input_channels = board_size * board_size + 1
        self.embedding_size = 8
        self.embedding = nn.Embedding(num_piece_types, self.embedding_size)
        
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
    
    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float).to("cpu").unsqueeze(0)
        x = self.embedding(x.long())
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
        
        return F.log_softmax(policy_logits, dim=1), value