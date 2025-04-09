import os
import argparse
from triple_town_simulate import TripleTownSim
from agent import TripleTownAgent
from model import TripleTownNet, TripleTownPredict
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 初始化環境和模型
    env = TripleTownSim()
    policy = TripleTownNet(device, board_size=6, num_piece_types=22)
    pridict = TripleTownPredict(device, action_space = 36, board_size=6, num_piece_types=22)

    
    # 載入現有模型（如果有）
    if os.path.exists("models/triple_town_policy_final.pt"):
        policy.load_state_dict(torch.load("models/triple_town_policy_final.pt"))
    if os.path.exists("models/triple_town_pridict_final.pt"):
        pridict.load_state_dict(torch.load("models/triple_town_pridict_final.pt"))
    
    # 訓練智能體
    trainer = TripleTownAgent(policy, pridict, device, env)
    trainer.train(episodes=1000, MCTS_depth=100)
    
    # 保存最終模型
    torch.save(policy.state_dict(), "models/triple_town_policy_final.pt")
    torch.save(pridict.state_dict(), "models/triple_town_pridict_final.pt")

if __name__ == "__main__":
    main()