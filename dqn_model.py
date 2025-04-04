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
    """Triple Town遊戲的深度Q網絡模型"""
    def __init__(self, board_size=6, embedding_dim=8):
        super(TripleTownDQN, self).__init__()
        # 基本參數
        self.board_size = board_size
        self.num_actions = board_size * board_size  # 6x6棋盤 = 36個可能的放置位置
        self.num_item_types = 22  # 0-21的物品ID
        self.embedding_dim = embedding_dim
        
        # 計算輸入大小 - 棋盤(6x6)和下一個物品(6x6)
        self.input_cells = board_size * board_size + 1
        
        # 物品嵌入層 - 將物品ID轉換為向量表示
        self.item_embedding = nn.Embedding(self.num_item_types, self.embedding_dim)
        
        # 計算嵌入後的特徵維度
        embedded_dim = self.input_cells * self.embedding_dim
        
        # 定義神經網絡層 - 在初始化時建立，而非每次前向傳播
        self.fc1 = nn.Linear(embedded_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.num_actions)
    
    def forward(self, state):
        """前向傳播處理遊戲狀態並輸出每個動作的Q值"""
        batch_size = state.size(0)
        
        # 將狀態展平為二維張量 [batch_size, cells]
        state_flat = state.view(batch_size, -1)
        
        # 確保輸入到Embedding層的是長整型
        state_long = state_flat.long()
        
        # 將物品ID轉換為向量表示
        item_embeddings = self.item_embedding(state_long)
        
        # 將嵌入後的向量展平 [batch_size, cells*embedding_dim]
        state_vector = item_embeddings.view(batch_size, -1)
        
        # 使用ReLU激活函數和批標準化的全連接層
        x = F.relu(self.fc1(state_vector))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 輸出層 - 每個動作的Q值
        q_values = self.fc4(x)
        
        return q_values