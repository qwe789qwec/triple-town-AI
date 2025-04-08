import torch.optim as optim
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from triple_town_simulate import TripleTownSim
import random
import time

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

    
class TripleTownRL:
    def __init__(self, net, env, lr=0.001, buffer_size=10000, batch_size=128):
        self.net = net
        self.env = env
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def policy_value_function(self, state):
        """包裝神經網絡輸出為MCTS可用的格式"""
        log_probs, value = self.net(state)
        return log_probs, value.item()
    
    def collect_self_play_data(self, n_games=1):
        """通過自我對局收集訓練數據"""
        data = []
        
        for _ in range(n_games):
            mcts = MCTS(self.policy_value_function, n_simulations=400)
            state = self.env.reset()
            
            states, mcts_probs, rewards = [], [], []
            action = None

            while True:
                # 執行MCTS搜索
                mcts.search(self.env)
                
                # 根據訪問計數獲取行動概率
                action_probs = mcts.get_action_probs(state, temp=1.0)
                
                # 保存當前狀態與MCTS概率
                states.append(state)
                mcts_probs.append(action_probs)
                
                # 依概率選擇動作
                action = np.random.choice(len(action_probs), p=action_probs)

                # 執行動作
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                
                if done:
                    # 處理累積獎勵
                    returns = np.zeros_like(rewards, dtype=np.float32)
                    cumulative_return = 0
                    for t in reversed(range(len(rewards))):
                        cumulative_return = rewards[t] + 0.99 * cumulative_return
                        returns[t] = cumulative_return
                    
                    # 儲存資料
                    for t in range(len(states)):
                        data.append((states[t], mcts_probs[t], returns[t]))
                    
                    break
        
        return data
    
    def train_step(self):
        """執行一次訓練步驟"""
        if len(self.buffer) < self.batch_size:
            return None
        
        # 從經驗回放緩衝區取樣
        minibatch = random.sample(self.buffer, self.batch_size)
    
        # 逐個處理狀態
        all_log_probs = []
        all_values = []
        
        for data in minibatch:
            state = data[0]
            log_prob, value = self.net(state)
            all_log_probs.append(log_prob)
            all_values.append(value)
        
        # 堆疊結果
        log_probs = torch.cat(all_log_probs, dim=0)
        values = torch.cat(all_values, dim=0)
        
        # 將列表轉換為 numpy 數組，然後轉為張量 (避免警告)
        mcts_probs_batch = torch.FloatTensor(np.array([data[1] for data in minibatch])).to(log_probs.device)
        returns_batch = torch.FloatTensor(np.array([data[2] for data in minibatch])).unsqueeze(1).to(values.device)
        
        # 計算損失
        value_loss = F.mse_loss(values, returns_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_probs, dim=1))
        total_loss = value_loss + policy_loss
        
        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def train(self, n_epochs=100, games_per_epoch=10):
        """完整訓練流程"""
        for epoch in range(n_epochs):
            # 收集自我對局數據
            data = self.collect_self_play_data(n_games=games_per_epoch)
            self.buffer.extend(data)
            
            # 訓練模型
            avg_loss = 0
            for _ in range(5):  # 每輪進行多步訓練
                loss = self.train_step()
                if loss is not None:
                    avg_loss += loss / 5
            
            print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
            
            # 定期保存模型
            if (epoch + 1) % 10 == 0:
                torch.save(self.net.state_dict(), f"triple_town_model_epoch_{epoch+1}.pt")
