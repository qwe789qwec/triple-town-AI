import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# 定義經驗元組
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """經驗回放緩衝區"""
    def __init__(self, capacity=100000):
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
    """深度Q網絡模型"""
    def __init__(self):
        super(TripleTownDQN, self).__init__()
        # 基本參數
        self.board_size = 6
        self.num_item_types = 22  # 0-21的物品ID
        self.embedding_dim = 32  # 物品嵌入維度
        
        # 物品嵌入層 - 將物品ID轉換為向量表示
        self.item_embedding = nn.Embedding(self.num_item_types, self.embedding_dim)
        
        # 遊戲板處理 - 使用卷積神經網絡提取特徵
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 當前物品處理 - 使用全連接層
        self.item_fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
        )
        
        # 合併後的全連接層
        self.combined_fc = nn.Sequential(
            nn.Linear(64 * self.board_size * self.board_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 36)  # 36個輸出 - 對應36個可能的動作
        )
    
    def forward(self, state):
        """前向傳播"""
        batch_size = state.size(0)
        
        # 分離當前物品和遊戲板
        current_item = state[:, 0].long()
        board = state[:, 1:].reshape(batch_size, self.board_size, self.board_size).long()
        
        # 處理當前物品
        item_embedded = self.item_embedding(current_item)
        item_features = self.item_fc(item_embedded)
        
        # 處理遊戲板
        board_embedded = self.item_embedding(board)  # [batch, 6, 6, embed_dim]
        board_embedded = board_embedded.permute(0, 3, 1, 2)  # [batch, embed_dim, 6, 6]
        board_features = self.conv_layers(board_embedded)
        # board_features = board_features.view(batch_size, -1)  # 展平
        board_features = board_features.reshape(batch_size, -1)  # 展平
        
        # 合併特徵
        combined_features = torch.cat([board_features, item_features], dim=1)
        
        # 計算Q值
        q_values = self.combined_fc(combined_features)
        
        return q_values