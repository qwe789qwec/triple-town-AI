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
    
class MCTSNode:
    def __init__(self, prior_prob=0, parent=None):
        self.parent = parent
        self.children = {}  # 子節點: {action: MCTSNode}
        self.visit_count = 0  # 訪問次數
        self.value_sum = 0.0  # 累積價值
        self.prior_prob = prior_prob  # 先驗概率
        
    def expand(self, actions_probs):
        """根據策略網絡的輸出擴展節點"""
        for action, prob in actions_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(prior_prob=prob, parent=self)
    
    def select(self, c_puct=5.0):
        """選擇最有價值的子節點"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        # UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        for action, child in self.children.items():
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
                u_value = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
                score = q_value + u_value
            else:
                # 優先訪問未探索的節點
                score = c_puct * child.prior_prob * math.sqrt(self.visit_count + 1e-8)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def update(self, value):
        """更新節點統計信息"""
        self.visit_count += 1
        self.value_sum += value
    
    def is_leaf(self):
        """檢查是否為葉節點"""
        return len(self.children) == 0
    
    def is_root(self):
        """檢查是否為根節點"""
        return self.parent is None

class MCTS:
    def __init__(self, policy_value_fn, n_simulations=1000, c_puct=5.0):
        self.policy_value_fn = policy_value_fn
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.root = MCTSNode()
    
    def search(self, env):
        """執行MCTS搜索"""
        for _ in range(self.n_simulations):
            # 模擬環境用於搜索
            sim_env = env.copy()  # 假設環境支援複製
            sim_state = sim_env.now_state

            # 階段1: 選擇
            node = self.root
            search_path = [node]
            action_path = []
            done = False


            # 選擇階段: 尋找葉節點
            while not node.is_leaf():
                action, node = node.select(self.c_puct)
                search_path.append(node)
                action_path.append(action)

                sim_state, reward, done, _ = sim_env.step(action)
                if done:
                    break

            # 階段2: 擴展與評估
            if not done:
                # 使用策略網絡評估該狀態
                log_probs, value = self.policy_value_fn(sim_state)
                probs = np.exp(log_probs.cpu().detach().numpy())
                valid_probs = []
                # 只考慮合法動作
                valid_moves = sim_env.get_valid_actions(sim_state)
                for a in range(len(valid_moves)):
                    if valid_moves[a] == 1:
                        valid_probs.append((a, probs[0][a]))
                
                # 擴展節點
                node.expand(valid_probs)
            else:
                # 遊戲結束，使用實際獎勵作為評估
                value = reward
            
            # 階段3: 回溯更新
            for node in reversed(search_path):
                node.update(value)
                # value = -value  # 對於單人遊戲，可以移除此行
    
    def get_action_probs(self, state, temp=1.0):
        """獲取行動概率分布"""
        # 根據訪問次數計算概率
        visit_counts = np.array([
            self.root.children[a].visit_count if a in self.root.children else 0
            for a in range(6 * 6)
        ])
        
        if temp == 0:  # 貪婪選擇
            action = np.argmax(visit_counts)
            probs = np.zeros_like(visit_counts)
            probs[action] = 1.0
            return probs
        else:
            # 溫度控制探索程度
            scaled_counts = visit_counts ** (1.0 / temp)
            probs = scaled_counts / np.sum(scaled_counts)
            return probs
    
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


def main():
    # 初始化環境和模型
    env = TripleTownSim()
    net = TripleTownNet(board_size=6, num_piece_types=22)
    
    # 載入現有模型（如果有）
    try:
        net.load_state_dict(torch.load("triple_town_model.pt"))
        print("加載預訓練模型")
    except:
        print("從頭開始訓練")
    
    # 訓練智能體
    trainer = TripleTownRL(net, env)
    trainer.train(n_epochs=100, games_per_epoch=10)
    
    # 保存最終模型
    torch.save(net.state_dict(), "triple_town_model_final.pt")

if __name__ == "__main__":
    main()