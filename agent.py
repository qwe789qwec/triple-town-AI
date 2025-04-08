import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from model import ReplayBuffer
from MCTS import MCTSNode, MCTS

class TripleTownAgent:
    """Triple Town智能體"""
    def __init__(self, net, env, lr=0.0003, buffer_size=10000, batch_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = net
        self.env = env
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = ReplayBuffer(capacity=500000)
        self.batch_size = batch_size

        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.9995  # 探索率衰減率
        self.epsilon_min = 0.1  # 最小探索率

    def select_action(self, state, block, MCTS_depth = 1000, explore=True):
        """選擇動作"""
        # 使用MCTS進行搜索
        mcts = MCTS(self.policy_net, depth=MCTS_depth)
        mcts.search(self.env)
        action_probs = mcts.get_action_probs(temperature=self.epsilon)
        action_mask = self.env.get_valid_actions(state, block)
        action_softmax = np.exp(action_probs * action_mask) / np.sum(action_probs * action_mask)

        if explore and random.random() < self.epsilon:
            return random.choice(self.env.get_valid_actions(state, block)), action_softmax
        
        if explore:
            # softmax 採樣
            action = np.random.choice(len(action_probs), p=action_softmax)
        else:
            action = np.argmax(action_softmax)
        
        return action, action_softmax

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_probs_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        state_batch = torch.tensor(state_batch).to(self.device)
        action_probs_batch = torch.tensor(action_probs_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).to(self.device)
        next_state_batch = torch.tensor(next_state_batch).to(self.device)
        done_batch = torch.tensor(done_batch).to(self.device)

        # 使用策略網絡計算當前狀態的 Q 值
        net_action_probs, net_reward = self.policy_net(state_batch)

        reward_loss = (reward_batch - net_reward) ** 2
        
        # 2. 策略损失: -π^T ln p
        policy_loss = -torch.sum(action_probs_batch * torch.log(net_action_probs + 1e-8), dim=1)
        
        # 3. L2正则化: c||θ||²
        l2_reg = 0
        for param in self.policy_net.parameters():
            l2_reg += torch.norm(param) ** 2

        # 4. 總損失
        loss = reward_loss.mean() + policy_loss.mean() + 0.01 * l2_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, MCTS_depth = 100, model_dir = "models"):
        scores = []
        avg_scores = []

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            action = None
            done = False
            final_reward = 0
            
            # 記錄整個 episode 的經驗
            episode_buffer = []

            while not done:
                if action == 0:
                    block = True
                else:
                    block = False
                action, action_probs = self.select_action(state, block=block, MCTS_depth=MCTS_depth, explore=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_buffer.append((state, action_probs, reward, next_state, done))
                state = next_state
                final_reward = reward

            # 將經驗添加到回放緩衝區，使用 λ-return
            for t in range(len(episode_buffer)):
                state, action, reward, next_state, done = episode_buffer[t]
                
                left_reward = final_reward - reward

                self.memory.push(state, action, left_reward, next_state, done)
            
            # 從經驗中學習
            self.optimize_model()
            
            # 記錄分數
            scores.append(self.env.game_score)
            avg_scores.append(np.mean(scores[-300:]) if len(scores) >= 300 else np.mean(scores))
            
            # 定期打印進度
            if episode % 100 == 0:
                best_score = np.max(scores)
                print(f"Episode {episode}, Best Score: {best_score}, Avg Score: {avg_scores[-1]:.2f}, Epsilon: {self.epsilon:.4f}")
            
            # 定期保存模型
            if episode % 100 == 0:
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
            state = self.env.reset()
            done = False
            action = None
            
            while not done:
                # 使用學習到的策略選擇動作
                if action == 0:
                    block = True
                else:
                    block = False
                action = self.select_action(state, block, MCTS_depth=100, explore=False)
                next_state, reward, done, _ = self.env.step(action)
            
            scores.append(reward)
        
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.epsilon = checkpoint['epsilon']