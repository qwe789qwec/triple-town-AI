import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ReplayBuffer
from MCTS import MCTSNode, MCTS
import time

class TripleTownAgent:
    """Triple Town智能體"""
    def __init__(self, policy, predict, device, env, lr=0.0003, buffer_size=10000, batch_size=512):
        self.device = device

        self.policy_net = policy
        self.predict_net = predict
        self.env = env
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.predict_opt = optim.Adam(self.predict_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = ReplayBuffer(capacity=500000)
        self.batch_size = batch_size

        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.9995  # 探索率衰減率
        self.epsilon_min = 0.1  # 最小探索率

    def select_action(self, state, block, MCTS_depth = 100, explore=True):
        """選擇動作"""
        # 使用MCTS進行搜索
        mcts = MCTS(self.policy_net, depth=MCTS_depth)
        mcts.search(self.env)
        try:
            action_probs = mcts.get_action_probs(temperature=self.epsilon)
            action_mask = self.env.get_valid_actions(state, block)
            action_softmax = np.exp(action_probs * action_mask) / np.sum(action_probs * action_mask)

            masked_logits = action_probs[action_mask > 0.5]
            masked_softmax = np.exp(masked_logits - np.max(masked_logits))
            masked_softmax /= np.sum(masked_softmax)
            indices = np.where(action_mask > 0.5)[0]
        except Exception as e:
            print(f"error: {e}")
            print("state:", state)
            print("action_probs:", action_probs)
            print("action_mask:", action_mask)
            print("action_softmax:", action_softmax)
            print("masked_logits:", masked_logits)
            print("masked_softmax:", masked_softmax)
            print("indices:", indices)
            exit()

        if explore and random.random() < self.epsilon:
            return np.random.choice(indices), action_softmax
        
        if explore:
            # softmax 採樣
            action = np.random.choice(indices, p=masked_softmax)
        else:
            action = np.argmax(action_softmax)
        
        return action, action_softmax

    def optimize_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
         # 抽樣批次經驗
        batch = self.memory.sample(self.batch_size)
        
        # 轉換為張量
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.long, device=self.device)
        action_probs_batch = torch.tensor(np.array(batch.action_probs), dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.float, device=self.device)

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

        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
    
    def optimize_predict(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 抽樣批次經驗
        batch = self.memory.sample(self.batch_size)
        
        # 轉換為張量
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.long, device=self.device)
        action_probs_batch = torch.tensor(np.array(batch.action_probs), dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.float, device=self.device)
        zero_action = torch.zeros(len(action_batch[0]), dtype=torch.float, device=self.device)

        predict_state , state_embedding = self.predict_net(state_batch, action_batch)
        _, next_state_embedding = self.policy_net(next_state_batch, zero_action)
        predict_loss = (predict_state - next_state_embedding) ** 2

        self.predict_opt.zero_grad()
        predict_loss.mean().backward()
        self.predict_opt.step()


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
            self.optimize_policy()
            
            # 記錄分數
            print("final_reward", final_reward)
            scores.append(final_reward)
            avg_scores.append(np.mean(scores[-300:]) if len(scores) >= 300 else np.mean(scores))
            
            if episode % 10 == 0:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

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
            'optimizer': self.policy_opt.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """載入模型"""
        print(f"Loading model from {filename}")
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
