import os
import argparse
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent
from model import TripleTownNet
import torch

def main():
    # 初始化環境和模型
    env = TripleTownSim()
    net = TripleTownNet(board_size=6, num_piece_types=22)
    
    # 載入現有模型（如果有）
    try:
        net.load_state_dict(torch.load("triple_town_model.pt"))
        print("no model")
    except:
        print("train new model")
    
    # 訓練智能體
    trainer = TripleTownAgent(net, env)
    trainer.train(episodes=1000, MCTS_depth=400, games_per_epoch=10)
    
    # 保存最終模型
    torch.save(net.state_dict(), "triple_town_model_final.pt")

if __name__ == "__main__":
    main()