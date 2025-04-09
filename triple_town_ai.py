from triple_town_game import TripleTownHandler
from triple_town_simulate import TripleTownSim
from model import TripleTownNet
import torch.nn.functional as F
import numpy as np
import torch
from MCTS import MCTSNode, MCTS

def main():
    # game = TripleTownHandler()
    gameSim = TripleTownSim()
    game = TripleTownHandler()
    net = TripleTownNet()
    net.load_state_dict(torch.load("models/triple_town_model_ep300.pt"))
    net.eval()
    # game.reset()
    action = -1
    # if game.screen_center.x == -1:
    #     exit()
    reward_list = []

    for i in range(200):
        state = game.game_status()
        done = False
        while not done:
            # state = game.game_status()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = net(state_tensor)
            gameSim.display_board(state)
            if action == 0:
                block = True
            else:
                block = False
            mask = gameSim.get_valid_actions(state, block)
            with torch.no_grad():  # 完全禁用梯度計算
                masked_logits = action_probs * mask
                # 將無效動作設為非常小的值（等同於-∞）
                masked_logits = masked_logits + (mask - 1) * 1e9
                probs = F.softmax(masked_logits, dim=1)
                
                # 方法1：直接使用PyTorch採樣
                # m = torch.distributions.Categorical(probs)
                # action = m.sample().item()

                # 方法2：或轉換為NumPy後採樣
                probs_np = probs.squeeze(0).cpu().numpy()
                action = np.random.choice(len(probs_np), p=probs_np)
                
                # 方法3：或使用numpy的argmax
                # action = np.argmax(probs.squeeze(0).cpu().numpy())

            new_state, reward, done, _ = game.step(action)
            # print("action: ", action)
            # game.click_slot(action)
            state = new_state
            if gameSim.is_game_over(state):
                if reward > 100:
                    print("reward: ", reward)
                    print("game over")
                reward_list.append(reward)
    
    print("average reward: ", np.mean(reward_list))

def main_sim():
    gameSim = TripleTownSim()
    net = TripleTownNet()
    net.load_state_dict(torch.load("models/triple_town_model_ep300.pt"))
    net.eval()
    # game.reset()
    action = -1
    # if game.screen_center.x == -1:
    #     exit()
    reward_list = []

    for i in range(200):
        state = gameSim.reset()
        done = False
        while not done:
            # state = game.game_status()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs, _ = net(state_tensor)
            # gameSim.display_board(state)
            if action == 0:
                block = True
            else:
                block = False
            mask = gameSim.get_valid_actions(state, block)
            with torch.no_grad():  # 完全禁用梯度計算
                masked_logits = action_probs * mask
                # 將無效動作設為非常小的值（等同於-∞）
                masked_logits = masked_logits + (mask - 1) * 1e9
                probs = F.softmax(masked_logits, dim=1)
                
                # 方法1：直接使用PyTorch採樣
                # m = torch.distributions.Categorical(probs)
                # action = m.sample().item()

                # 方法2：或轉換為NumPy後採樣
                probs_np = probs.squeeze(0).cpu().numpy()
                action = np.random.choice(len(probs_np), p=probs_np)
                
                # 方法3：或使用numpy的argmax
                # action = np.argmax(probs.squeeze(0).cpu().numpy())

            new_state, reward, done, _ = gameSim.step(action)
            # print("action: ", action)
            # game.click_slot(action)
            state = new_state
            if gameSim.is_game_over(state):
                if gameSim.game_score > 100:
                    print("score: ", gameSim.game_score)
                    print("game over")
                reward_list.append(gameSim.game_score)
    
    print("average reward: ", np.mean(reward_list))

if __name__ == "__main__":
    main_sim()