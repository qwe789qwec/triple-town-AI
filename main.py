import os
import argparse
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent
from model import TripleTownNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 初始化環境和模型
    env = TripleTownSim()
    net = TripleTownNet(device, board_size=6, num_piece_types=22)
    
    # 訓練智能體
    trainer = TripleTownAgent(net, device, env)
    try:
        trainer.load("models/triple_town_model_ep200.pt")
    except:
        print("train new model")
        pass
    trainer.train(episodes=1000, MCTS_depth=1000)
    
    # 保存最終模型
    torch.save(net.state_dict(), "triple_town_model_final.pt")

if __name__ == "__main__":
    main()