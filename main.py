import os
import argparse
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent
from model import TripleTownNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN = False
VALIDATE = True
REALPLAY = False

def main():
    # 初始化環境和模型
    env = TripleTownSim()
    net = TripleTownNet(device, board_size=6, num_piece_types=22)
    model_path = "models/triple_town_model_ep400.pt"
    
    # 訓練智能體
    agent = TripleTownAgent(net, device, env)
    if os.path.exists(model_path):
        agent.load(model_path)

    if TRAIN:
        agent.train(episodes=1000, MCTS_depth=300)

        # 保存最終模型
        torch.save(net.state_dict(), "triple_town_model_final.pt")

    if VALIDATE:
        agent.validate(episodes=20, MCTS_depth=300)

    if REALPLAY:
        agent.realplay(MCTS_depth=300)

if __name__ == "__main__":
    main()