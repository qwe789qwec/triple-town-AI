import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os

from dqn_model import TripleTownDQN, ReplayBuffer, Experience

class TripleTownAgent:
    """Triple Town智能體"""
    def __init__(self, game, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.game = game
        self.device = device
        
        # 初始化模型
        self.policy_net = TripleTownDQN().to(device)
        self.target_net = TripleTownDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 設置優化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # 經驗回放
        self.memory = ReplayBuffer(capacity=100000)
        
        # 學習參數
        self.batch_size = 500
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.999995  # 探索率衰減
        self.target_update = 50  # 目標網絡更新頻率
        self.learn_counter = 0
    
    def state_to_tensor(self, state):
        """將狀態轉換為張量"""
        return torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
    
    def select_action(self, state, block = False, explore=True):
        """選擇動作 - epsilon-greedy策略"""
        valid_mask = self.game.get_valid_actions(state, block)
        valid_actions = np.where(valid_mask == 1)[0]
        
        # 探索: 隨機選擇有效動作
        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 利用: 選擇最佳Q值的有效動作
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # 遮罩無效動作
            masked_q_values = np.full(36, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            
            return np.argmax(masked_q_values)
    
    def optimize_model(self):
        """從回放緩衝區學習"""
        if len(self.memory) < self.batch_size:
            return
        
        # 抽樣批次經驗
        batch = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*batch))
        
        # 轉換為張量
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
        
        # 處理next_state (可能包含None)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                    dtype=torch.bool).to(self.device)
        
        valid_states = [s for s in batch.next_state if s is not None]
        if valid_states:
            valid_states_array = np.array(valid_states)
            non_final_next_states = torch.tensor(valid_states_array, dtype=torch.float).to(self.device)
        else:
            non_final_next_states = torch.zeros((0, state_batch.shape[1]), dtype=torch.float).to(self.device)
        
        # 計算當前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 計算目標Q值
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        target_q_values = reward_batch + (self.gamma * next_state_values)
        
        # 計算Huber損失
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # 優化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 更新目標網絡
        self.learn_counter += 1
        if self.learn_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    
    def save(self, filename):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """載入模型"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']