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
    def __init__(self, state=None, action_from_parent=None, prior_prob=0, parent=None):
        self.state = state  # 存儲當前狀態
        self.action_from_parent = action_from_parent  # 從父節點到達此節點的動作
        self.parent = parent
        self.children = {}  # {action: [MCTSNode, ...]} - 每個動作可能有多個後續狀態
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
    def expand(self, actions_probs, sim_env, n_samples=3):
        """根據策略網絡和環境模擬擴展節點"""
        for action, prob in actions_probs:
            if action not in self.children:
                self.children[action] = []
            
            # 為每個動作模擬多個可能的後續狀態
            for _ in range(n_samples):
                env_copy = TripleTownSim(sim_env)
                next_state, reward, done, _ = env_copy.step(action)
                
                # 為每個(action, next_state)對創建一個節點
                child_node = MCTSNode(
                    state=next_state,
                    action_from_parent=action,
                    prior_prob=prob/n_samples,  # 分配概率
                    parent=self
                )
                self.children[action].append(child_node)
    
    def select(self, c_puct=5.0):
        """選擇最有價值的子節點"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child_nodes in self.children.items():
            for child in child_nodes:
                if child.visit_count > 0:
                    q_value = child.value_sum / child.visit_count
                    u_value = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
                    score = q_value + u_value
                else:
                    score = c_puct * child.prior_prob * math.sqrt(self.visit_count + 1e-8)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
                    
        return best_action, best_child
    
    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, policy_value_fn, n_simulations=1000, c_puct=5.0, n_samples=3):
        self.policy_value_fn = policy_value_fn
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.n_samples = n_samples
        self.root = MCTSNode()
    
    def search(self, state):
        """執行MCTS搜索"""
        self.root.state = state
        
        for _ in range(self.n_simulations):
            node = self.root
            search_path = [node]
            sim_env = TripleTownSim(state)
            current_state = sim_env.now_state.copy()
            done = False
            
            # 階段1: 選擇
            while not node.is_leaf() and not done:
                action, node = node.select(self.c_puct)
                search_path.append(node)
                current_state = node.state
                
                # 如果不是根節點，更新模擬環境狀態
                if node.action_from_parent is not None:
                    _, reward, done, _ = sim_env.step(node.action_from_parent)
            
            # 階段2: 擴展與評估
            if not done:
                log_probs, value = self.policy_value_fn(current_state)
                probs = np.exp(log_probs.cpu().detach().numpy())
                valid_probs = []
                valid_moves = sim_env.get_valid_actions(current_state)
                for a in range(len(valid_moves)):
                    if valid_moves[a] == 1:
                        valid_probs.append((a, probs[0][a]))
                
                # 擴展節點，考慮每個動作的多種可能結果
                node.expand(valid_probs, current_state, self.n_samples)
            else:
                value = reward
            
            # 階段3: 回溯更新
            for node in reversed(search_path):
                node.update(value)
    
    def get_action_probs(self, state, temp=1.0):
        """獲取動作概率"""
        self.search(state)
        
        # 計算每個動作的總訪問次數
        action_visits = {}
        for action, child_nodes in self.root.children.items():
            action_visits[action] = sum(child.visit_count for child in child_nodes)
        
        actions = list(action_visits.keys())
        visits = np.array([action_visits[a] for a in actions])
        
        if temp == 0:  # 確定性選擇
            action_idx = np.argmax(visits)
            action_probs = np.zeros(len(actions))
            action_probs[action_idx] = 1.0
            return actions, action_probs
        else:
            # 根據溫度參數計算概率
            visits = visits ** (1.0 / temp)
            action_probs = visits / np.sum(visits)
            return actions, action_probs
    
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
                mcts.search(state)
                
                # 根據訪問計數獲取行動概率
                actions, action_probs = mcts.get_action_probs(state, temp=1.0)
                
                # 保存當前狀態與MCTS概率
                states.append(state)
                mcts_probs.append(action_probs)
                
                # 依概率選擇動作
                action_idx = np.random.choice(len(action_probs), p=action_probs)
                action = actions[action_idx]

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

        # 方法1：從環境獲取
        max_actions = self.env.action_space.n
        # 方法2：在訓練過程中動態確定
        max_len = max(len(data[1]) for data in minibatch)

        # 逐個處理狀態
        all_log_probs = []
        all_values = []
        mcts_probs_list = []
        returns_list = []
        
        for data in minibatch:
            state = data[0]
            macts_probs = data[1]
            returns_value = data[2]

            padded_probs = np.zeros(max_len)
            padded_probs[:len(macts_probs)] = macts_probs

            log_prob, value = self.net(state)
            all_log_probs.append(log_prob)
            all_values.append(value)
            mcts_probs_list.append(padded_probs)
            returns_list.append(returns_value)
        
        # 堆疊結果
        log_probs = torch.cat(all_log_probs, dim=0)
        values = torch.cat(all_values, dim=0)

        # 將列表轉換為 numpy 數組，然後轉為張量 (避免警告)
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_list)).to(log_probs.device)
        returns_batch = torch.FloatTensor(np.array(returns_list)).unsqueeze(1).to(values.device)
        
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