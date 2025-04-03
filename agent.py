import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from triple_town_simulate import TripleTownSim

from dqn_model import TripleTownDQN, ReplayBuffer, Experience

class TripleTownAgent:
    """Triple Town智能體"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.game = TripleTownSim()
        self.device = device
        
        # 初始化模型
        self.policy_net = TripleTownDQN().to(device)
        self.target_net = TripleTownDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 設置優化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        
        # 經驗回放
        self.memory = ReplayBuffer(capacity=500000)
        
        # 學習參數
        self.batch_size = 1000
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.9995  # 探索率衰減
        self.target_update = 50  # 目標網絡更新頻率
        self.learn_counter = 0

        # game params
        self.item_list = np.zeros(len(self.game.ITEMS))
    
    def select_action(self, state, block = False, explore=True):
        """選擇動作 - epsilon-greedy策略"""
        valid_mask = self.game.get_valid_actions(state, block)
        valid_actions = np.where(valid_mask == 1)[0]
        
        # 探索: 隨機選擇有效動作
        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 利用: 選擇最佳Q值的有效動作
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
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
    
    def calculate_reward(self, prev_state, next_state, done):
        """改進的獎勵函數"""
        prev_item_count = np.zeros(len(self.game.ITEMS))
        next_item_count = np.zeros(len(self.game.ITEMS))
        
        if done:
            return -100  # 增加遊戲結束的懲罰
        
        if next_state is None:
            return -100  # 增加無效動作的懲罰
        
        # 分析狀態變化
        prev_board, _ = self.game._split_state(prev_state)
        next_board, _ = self.game._split_state(next_state)
        
        # 基礎獎勵
        reward = 0

        rows, cols = next_board.shape  # 假設是numpy陣列

        # 物品獎勵
        for row in range(rows):
            for col in range(cols):
                item = next_board[row, col]
                prev_item_count[prev_board[row, col]] += 1
                next_item_count[next_board[row, col]] += 1
                if self.item_list[item] == 0:
                    self.item_list[item] = 1
                    if(item < 10):
                        reward = item * 5
                    else:
                        reward = (item - 10) * 10
                    if(item >= 5 and item <= 9):
                        self.game.display_board(next_state)

        # 物品獎勵
        for item in range(len(self.game.ITEMS)):
            if item in [10, 11, 12, 19]:
                continue
            if prev_item_count[item] < next_item_count[item]:
                if item < 10:
                    reward += item * 2
                else:
                    reward += (item - 10) * 3

        # 空間管理獎勵
        empty_prev = np.sum(prev_board == 0)
        empty_next = np.sum(next_board == 0)
        if empty_next > empty_prev:
            reward += 3
        
        return reward

    def train(self, episodes, model_dir = "models"):
        scores = []
        avg_scores = []
        total_reward = 0

        for episode in tqdm(range(episodes)):
            state = self.game.reset()
            action = None
            done = False
            self.item_list = np.zeros(len(self.game.ITEMS))

            while not done:
                # 選擇並執行動作
                if action == 0:
                    block = True
                else:
                    block = False
                action = self.select_action(state, block)
                next_state = self.game.next_state(state, action)
                
                # 檢查遊戲是否結束
                if next_state is None or self.game.is_game_over(next_state):
                    done = True
                
                # 計算獎勵
                reward = self.calculate_reward(state, next_state, done)
                total_reward += reward

                # 存儲經驗
                self.memory.push(state, action, reward, next_state if not done else None, done)
                
                # 從經驗中學習
                self.optimize_model()
                
                # 更新狀態
                if not done and next_state is not None:
                    state = next_state
            
            # 記錄分數
            scores.append(self.game.game_score)
            avg_scores.append(np.mean(scores[-300:]) if len(scores) >= 300 else np.mean(scores))
            
            # 定期打印進度
            if episode % 300 == 0:
                best_score = np.max(scores)
                print(f"Episode {episode}, Best Score: {best_score}, Avg Score: {avg_scores[-1]:.2f}, Epsilon: {self.epsilon:.4f}")
            
            # 定期保存模型
            if episode % 1000 == 0:
                self.save(f"{model_dir}/triple_town_model_ep{episode}.pt")
        
        # 保存最終模型
        self.save(f"{model_dir}/triple_town_model_final.pt")
        
        # 繪製學習曲線
        plt.figure(figsize=(10, 6))
        plt.plot(scores, alpha=0.3)
        plt.plot(avg_scores, linewidth=2)
        plt.title('Triple Town Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig(f'{model_dir}/triple_town_learning.png')
        plt.close()
    
    def validate(self, episodes = 20):
        scores = []
    
        for i in range(episodes):
            state = self.game.reset()
            done = False
            action = None
            
            while not done:
                # 使用學習到的策略選擇動作
                if action == 0:
                    block = True
                else:
                    block = False
                action = self.select_action(state, block, explore=False)
                next_state = self.game.next_state(state, action)

                if next_state is None or self.game.is_game_over(next_state):
                    done = True
                else:
                    state = next_state
            
            scores.append(self.game.game_score)
        
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        print(f"\nEvaluation Results:")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Maximum Score: {max_score}")

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
        # self.epsilon = checkpoint['epsilon']