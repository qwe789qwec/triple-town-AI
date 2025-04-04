import os
import argparse
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent

TRAIN = True
EVAL = True

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="Triple Town DQN Training")
    parser.add_argument("--episodes", type=int, default=10000, help="訓練回合數")
    args = parser.parse_args()
    
    # 訓練
    if TRAIN:
        agent = TripleTownAgent()
        agent.train(args.episodes)
    
    # 只評估不訓練
    if EVAL:
        agent = TripleTownAgent()
        
        # 載入模型進行評估
        model_path = "models/triple_town_model_final.pt"
        if os.path.exists(model_path):
            agent.load(model_path)
            print(f"已載入模型: {model_path}")
            agent.validate(100)
        else:
            print(f"找不到模型: {model_path}。請先訓練或指定有效的模型路徑。")

        

if __name__ == "__main__":
    main()