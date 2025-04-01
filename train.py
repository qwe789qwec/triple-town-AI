import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
    
    # 直接獎勵合併事件 - 每次成功合併給予明確獎勵
    for item_id in range(3, game.ITEMS["castle"] + 1):  # 從草開始往上計算
        prev_count = np.sum(prev_board == item_id)
        next_count = np.sum(next_board == item_id)
        if next_count > prev_count:
            # 物品等級越高，獎勵指數增長
            reward += (next_count - prev_count) * (2 ** (item_id - 2))
    
    # 空間管理獎勵
    empty_prev = np.sum(prev_board == 0)
    empty_next = np.sum(next_board == 0)
    if empty_next > empty_prev:
        reward += 3  # 鼓勵清理空間
    elif empty_next < 10:  # 空間過少時的懲罰
        reward -= 2 * (10 - empty_next)  # 空間越少懲罰越大
    
    # 策略獎勵 - 獎勵創造潛在合併機會
    potential_reward = 0
    for i in range(game.board_size):
        for j in range(game.board_size):
            if next_board[i, j] > 0:  # 非空格
                # 檢查相鄰位置是否有相同物品
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < game.board_size and 0 <= nj < game.board_size:
                        if next_board[i, j] == next_board[ni, nj]:
                            potential_reward += 2  # 發現一對相同物品
    
    prev_board[0, 0] = 0
    next_board[0, 0] = 0
    if np.array_equal(prev_board, next_board):
        potential_reward -= 1
    
    reward += potential_reward
    
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