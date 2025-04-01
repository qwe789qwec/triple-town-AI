import os
import argparse
import time
import torch
import torch.multiprocessing as mp
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent
from train import train_agent, evaluate_agent, calculate_reward
from dqn_model import TripleTownDQN, Experience

def worker_process(
    worker_id, 
    shared_model, 
    experience_queue, 
    episode_scores,
    args, 
    device_id=None
):
    """工作進程負責運行遊戲並收集經驗"""
    # 設定裝置，如果指定了GPU ID就使用，否則使用CPU
    device = f"cuda:{device_id}" if device_id is not None and torch.cuda.is_available() else "cpu"
    
    # 創建本地遊戲環境
    game = TripleTownSim()
    
    # 創建本地智能體，但使用共享模型的參數
    local_agent = TripleTownAgent(game, device=device)
    
    # 主循環：收集經驗
    episodes_completed = 0
    while episodes_completed < args.worker_episodes:
        # 同步本地模型與共享模型
        local_agent.policy_net.load_state_dict(shared_model.state_dict())
        
        # 重置遊戲環境
        state = game.reset()
        done = False
        episode_reward = 0
        
        # 執行一個完整的遊戲場景
        while not done:
            # 選擇動作
            action = local_agent.select_action(state)
            next_state = game.next_state(state, action)
            
            # 檢查遊戲是否結束
            if next_state is None or game.is_game_over(next_state):
                done = True
            
            # 計算獎勵
            reward = calculate_reward(game, state, next_state, done)
            episode_reward += reward
            
            # 將經驗加入隊列，讓主進程進行學習
            experience_queue.put(
                (state, action, reward, next_state if not done else None, done)
            )
            
            # 更新狀態
            if not done and next_state is not None:
                state = next_state
        
        # 記錄分數
        episode_scores.put((worker_id, game.game_score))
        episodes_completed += 1
        
        # 定期報告進度
        if episodes_completed % 10 == 0:
            print(f"Worker {worker_id}: Completed {episodes_completed} episodes")

def learner_process(
    shared_model, 
    experience_queue, 
    episode_scores,
    args,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """學習進程負責從經驗中學習並更新共享模型"""
    # 創建遊戲環境用於評估（不實際使用來玩遊戲）
    game = TripleTownSim()
    
    # 創建智能體
    agent = TripleTownAgent(game, device=device)
    agent.policy_net.load_state_dict(shared_model.state_dict())
    
    # 如果已經有現有模型，先載入
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        shared_model.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded model from {args.load_model}")
    elif not args.load_model and os.path.exists(f"{args.model_dir}/triple_town_model_final.pt"):
        agent.load(f"{args.model_dir}/triple_town_model_final.pt")
        shared_model.load_state_dict(agent.policy_net.state_dict())
        print(f"Loaded latest model from {args.model_dir}/triple_town_model_final.pt")
    
    # 訓練統計
    episodes_completed = 0
    all_scores = []
    optimization_steps = 0
    last_save_time = time.time()
    
    # 創建儲存目錄
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 主循環：從經驗隊列學習
    while True:
        # 檢查是否已經達到訓練目標
        if episodes_completed >= args.episodes:
            print(f"Completed {args.episodes} episodes. Training finished.")
            break
        
        # 收集經驗
        try:
            experience = experience_queue.get(timeout=1.0)
            agent.memory.push(*experience)
        except:
            # 隊列為空，繼續檢查其他事項
            pass
        
        # 嘗試讀取完成的遊戲分數
        try:
            while True:
                worker_id, score = episode_scores.get_nowait()
                all_scores.append(score)
                episodes_completed += 1
                
                # 每100個episode打印統計信息
                if len(all_scores) % 100 == 0:
                    avg_score = sum(all_scores[-100:]) / 100
                    print(f"Episodes: {episodes_completed}, Avg Score (last 100): {avg_score:.2f}")
        except:
            # 分數隊列為空，繼續前進
            pass
        
        # 如果有足夠經驗，進行模型優化
        if len(agent.memory) >= agent.batch_size:
            agent.optimize_model()
            optimization_steps += 1
            
            # 每100步更新共享模型
            if optimization_steps % 100 == 0:
                shared_model.load_state_dict(agent.policy_net.state_dict())
        
        # 每隔一段時間保存模型
        current_time = time.time()
        if current_time - last_save_time > 300:  # 每5分鐘
            agent.save(f"{args.model_dir}/triple_town_model_step{optimization_steps}.pt")
            last_save_time = current_time
    
    # 保存最終模型
    agent.save(f"{args.model_dir}/triple_town_model_final.pt")
    
    # 如果需要評估
    if args.eval:
        print(f"\n=== Evaluating agent over {args.eval_games} games ===")
        evaluate_agent(agent, game, num_games=args.eval_games)
    
    return all_scores

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="Triple Town DQN Training with Parallel Environments")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate the agent")
    parser.add_argument("--episodes", type=int, default=50000, help="Total number of training episodes")
    parser.add_argument("--worker-episodes", type=int, default=1000, help="Episodes per worker")
    parser.add_argument("--eval-games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for saving/loading models")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a specific model")
    parser.add_argument("--num-workers", type=int, default=30, help="Number of parallel workers")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training if available")
    
    args = parser.parse_args()
    
    # 如果既不訓練也不評估，預設都執行
    if not args.train and not args.eval:
        args.train = True
        args.eval = True
    
    if args.train:
        # 設置多進程支持
        mp.set_start_method('spawn', force=True)
        
        # 創建共享模型（用於所有工作進程）
        device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        shared_model = TripleTownDQN().to(device)
        shared_model.share_memory()  # 使模型參數在進程間共享
        
        # 創建進程間通信隊列
        experience_queue = mp.Queue(maxsize=10000)  # 經驗隊列
        episode_scores = mp.Queue()  # 分數隊列
        
        # 啟動多個工作進程
        workers = []
        for i in range(args.num_workers):
            # 如果有多個GPU，可以為每個工作進程指定不同的GPU
            gpu_id = i % torch.cuda.device_count() if args.use_gpu and torch.cuda.is_available() else None
            p = mp.Process(
                target=worker_process,
                args=(i, shared_model, experience_queue, episode_scores, args, gpu_id)
            )
            p.start()
            workers.append(p)
        
        # 啟動學習進程（在主進程中運行）
        scores = learner_process(shared_model, experience_queue, episode_scores, args)
        
        # 等待所有工作進程完成
        for p in workers:
            p.join()
        
        print("All processes completed.")
    
    # 如果只想評估而不訓練
    elif args.eval:
        game = TripleTownSim()
        agent = TripleTownAgent(game)
        
        # 載入模型進行評估
        model_path = args.load_model if args.load_model else f"{args.model_dir}/triple_town_model_final.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"Loaded model from {model_path}")
            evaluate_agent(agent, game, num_games=args.eval_games)
        else:
            print(f"No model found at {model_path}. Please train first or specify a valid model path.")

if __name__ == "__main__":
    main()