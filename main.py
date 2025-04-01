from triple_town_simulate import TripleTownSim
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
        # 經驗回放
        self.memory = ReplayBuffer(capacity=100000)
        
        # 學習參數
        self.batch_size = 64
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.999  # 探索率衰減
        self.target_update = 10  # 目標網絡更新頻率
        self.learn_counter = 0
    
    def state_to_tensor(self, state):
        """將狀態轉換為張量"""
        return torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
    
    def select_action(self, state, explore=True):
        """選擇動作 - epsilon-greedy策略"""
        valid_mask = self.game.get_valid_actions(state)
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
        
        # 處理next_state (可能包含None) - 優化版本
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
            if len(non_final_next_states) > 0:  # 確保有非終止狀態
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        target_q_values = reward_batch + (self.gamma * next_state_values)
        
        # 計算Huber損失
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # 優化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 - 防止梯度爆炸
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

def calculate_reward(game, prev_state, next_state, done):
    """設計獎勵函數"""
    if done:
        return -10  # 遊戲結束的懲罰
    
    if next_state is None:
        return -5  # 錯誤的動作
    
    # 基礎獎勵 - 分數增加
    base_reward = 1
    
    # 分析狀態變化
    prev_board, _ = game._split_state(prev_state)
    next_board, _ = game._split_state(next_state)
    
    # 計算各類物品數量變化
    reward = 0
    
    # 獎勵生成高級物品
    for item_id, item_name in game.ITEM_NAMES.items():
        prev_count = np.sum(prev_board == item_id)
        next_count = np.sum(next_board == item_id)
        
        # 高級物品獎勵（根據等級給予不同獎勵）
        if item_id >= game.ITEMS["hut"]:  # 從小屋開始算高級物品
            if next_count > prev_count:
                # 物品等級越高，獎勵越大
                reward += (next_count - prev_count) * (item_id * 1.5)
    
    # 獎勵空間管理 - 鼓勵保持足夠的空位
    empty_prev = np.sum(prev_board == 0)
    empty_next = np.sum(next_board == 0)
    if empty_next > empty_prev:
        reward += 2  # 清理出更多空間
    elif empty_next >= 15:  # 保持充足空間
        reward += 1
    
    # 熊管理獎勵
    bear_prev = np.sum(prev_board == game.ITEMS["bear"]) + np.sum(prev_board == game.ITEMS["Nbear"])
    bear_next = np.sum(next_board == game.ITEMS["bear"]) + np.sum(next_board == game.ITEMS["Nbear"])
    tombstone_prev = np.sum(prev_board == game.ITEMS["tombstone"])
    tombstone_next = np.sum(next_board == game.ITEMS["tombstone"])
    
    if bear_prev > bear_next and tombstone_next > tombstone_prev:
        reward += 3  # 成功將熊轉化為墓碑
    
    # 返回總獎勵
    return base_reward + reward

def train_agent(num_episodes=5000):
    """訓練智能體"""
    game = TripleTownSim()
    agent = TripleTownAgent(game)

    # if have model
    if os.path.exists("triple_town_model_final.pt"):
        agent.load("triple_town_model_final.pt")
        print("load model")
    
    # 記錄訓練過程
    scores = []
    avg_scores = []
    
    for episode in tqdm(range(num_episodes)):
        state = game.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 選擇並執行動作
            action = agent.select_action(state)
            next_state = game.next_state(state, action)
            
            # 檢查遊戲是否結束
            if next_state is None or game.is_game_over(next_state):
                done = True
            
            # 計算獎勵
            reward = calculate_reward(game, state, next_state, done)
            total_reward += reward
            
            # 存儲經驗
            agent.memory.push(state, action, reward, next_state if not done else None, done)
            
            # 從經驗中學習
            agent.optimize_model()
            
            # 更新狀態
            if not done and next_state is not None:
                state = next_state
        
        # 記錄分數
        scores.append(game.game_score)
        avg_scores.append(np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores))
        
        # 定期打印進度
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {game.game_score}, Avg Score: {avg_scores[-1]:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # 定期保存模型
        if episode % 500 == 0:
            agent.save(f"triple_town_model_ep{episode}.pt")
    
    # 保存最終模型
    agent.save("triple_town_model_final.pt")
    
    # 繪製學習曲線
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3)
    plt.plot(avg_scores, linewidth=2)
    plt.title('Triple Town Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('triple_town_learning.png')
    plt.close()
    
    return agent, scores

def evaluate_agent(agent, num_games=50):
    """評估智能體表現"""
    game = TripleTownSim()
    scores = []
    
    for i in range(num_games):
        state = game.reset()
        done = False
        
        while not done:
            # 使用學習到的策略選擇動作
            action = agent.select_action(state, explore=False)
            next_state = game.next_state(state, action)
            
            if next_state is None or game.is_game_over(next_state):
                done = True
            else:
                state = next_state
        
        scores.append(game.game_score)
        print(f"Game {i+1}: Score = {game.game_score}")
    
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Maximum Score: {max_score}")
    
    return scores

# 主函數
if __name__ == "__main__":
    # 訓練智能體
    trained_agent, training_scores = train_agent(num_episodes=5000)
    
    # 評估智能體表現
    evaluation_scores = evaluate_agent(trained_agent, num_games=50)