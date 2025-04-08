import math
import numpy as np

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
    def __init__(self, policy_value_fn, depth=1000, c_puct=5.0):
        self.policy_value_fn = policy_value_fn
        self.depth = depth
        self.c_puct = c_puct
        self.root = MCTSNode()
    
    def search(self, env):
        """執行MCTS搜索"""
        for _ in range(self.depth):
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
    
    def get_action_probs(self, temp=1.0):
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