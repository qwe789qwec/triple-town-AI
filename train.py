import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

item_list = {}

def calculate_reward(game, prev_state, next_state, done):
    """改進的獎勵函數"""
    if done:
        return -20  # 增加遊戲結束的懲罰
    
    if next_state is None:
        return -10  # 增加無效動作的懲罰
    
    # 分析狀態變化
    prev_board, _ = game._split_state(prev_state)
    next_board, _ = game._split_state(next_state)
    
    # 基礎獎勵
    reward = 0

    rows, cols = next_board.shape  # 假設是numpy陣列
    
    for row in range(rows):
        for col in range(cols):
            item = next_board[row, col]
            # 如果之前沒見過這種物品，給予獎勵
            if item not in item_list:
                item_list[item] = 1
                reward += 10
    
    # 空間管理獎勵
    empty_prev = np.sum(prev_board == 0)
    empty_next = np.sum(next_board == 0)
    # if empty_next > empty_prev:
    #     reward += 1

    # item connection 獎勵
    for row in range(rows):
        for col in range(cols):
            item = next_board[row, col]
            if item != 0:
                # 檢查上方
                if row > 0 and next_board[row - 1, col] == item:
                    reward += 1
                # 檢查下方
                if row < rows - 1 and next_board[row + 1, col] == item:
                    reward += 1
                # 檢查左方
                if col > 0 and next_board[row, col - 1] == item:
                    reward += 1
                # 檢查右方
                if col < cols - 1 and next_board[row, col + 1] == item:
                    reward += 1
    
    prev_board[0, 0] = 0
    next_board[0, 0] = 0
    if np.array_equal(prev_board, next_board):
        reward -= 1
    
    return reward

def train_agent(agent, game, num_episodes=5000, model_dir="models"):
    """訓練智能體"""
    # 建立儲存目錄
    os.makedirs(model_dir, exist_ok=True)
    
    # 記錄訓練過程
    scores = []
    avg_scores = []
    
    for episode in tqdm(range(num_episodes)):
        state = game.reset()
        action = None
        done = False
        
        while not done:
            # 選擇並執行動作
            if action == 0:
                block = True
            else:
                block = False
            action = agent.select_action(state, block)
            next_state = game.next_state(state, action)
            
            # 檢查遊戲是否結束
            if next_state is None or game.is_game_over(next_state):
                done = True
            
            # 計算獎勵
            reward = calculate_reward(game, state, next_state, done)
            
            # 存儲經驗
            agent.memory.push(state, action, reward, next_state if not done else None, done)
            
            # 從經驗中學習
            agent.optimize_model()
            
            # 更新狀態
            if not done and next_state is not None:
                state = next_state
        
        # 記錄分數
        scores.append(game.game_score)
        avg_scores.append(np.mean(scores[-300:]) if len(scores) >= 300 else np.mean(scores))
        
        # 定期打印進度
        if episode % 300 == 0:
            best_score = np.max(scores)
            print(f"Episode {episode}, Best Score: {best_score}, Avg Score: {avg_scores[-1]:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # 定期保存模型
        if episode % 5000 == 0:
            agent.save(f"{model_dir}/triple_town_model_ep{episode}.pt")
    
    # 保存最終模型
    agent.save(f"{model_dir}/triple_town_model_final.pt")
    
    # 繪製學習曲線
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.3)
    plt.plot(avg_scores, linewidth=2)
    plt.title('Triple Town Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(f'{model_dir}/triple_town_learning.png')
    plt.close()
    
    return scores

def evaluate_agent(agent, game, num_games=50):
    """評估智能體表現"""
    scores = []
    
    for i in range(num_games):
        state = game.reset()
        done = False
        action = None
        
        while not done:
            # 使用學習到的策略選擇動作
            if action == 0:
                block = True
            else:
                block = False
            action = agent.select_action(state, block, explore=False)
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