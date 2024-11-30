import math
import numpy as np
import random
import time
import cv2
import os
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import triple_town_model
from triple_town_game import playgame

BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class TripleTownAI:
    def __init__(self, broad_size=BROAD_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, tau = TAU, learning_rate = LR, memory_size=10000):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.memory = triple_town_model.ReplayMemory(memory_size)
        self.Transition = self.memory.Transition
        self.game = playgame()

        self.old_score = 1
        self.top_reward = 1

        self.policy_net = triple_town_model.DQN(broad_size).to(self.device)
        self.target_net = triple_town_model.DQN(broad_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

    def load_memory_process(self, load_size=150, skip=0):
        game_folder = 'gameplay'
        print("start load memory")
        image_files = sorted(
            [f for f in os.listdir(game_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda f: os.path.getmtime(os.path.join(game_folder, f))
        )
        current_state = None
        for i in range(len(image_files)-1):
            j = i + 1
            current = image_files[i]
            next = image_files[j]

            numbers1 = current.replace("game_", "").replace(".png", "").split("_")
            current_num1, current_step, current_action = map(int, numbers1)
            numbers2 = next.replace("game_", "").replace(".png", "").split("_")
            next_num2, next_step, next_action = map(int, numbers2)
            print(current)

            if current_num1 < skip:
                continue
            if current_num1 == next_num2 and next_step - current_step == 1:
                if current_state is None:
                    self.game.latest_image = cv2.imread(os.path.join(game_folder, current))
                    current_state, next_item = self.game.get_game_area()
                    all_state = self.game.slot_with_item(current_state, next_item)
                    current_score = self.game.get_score()
                    current_state_tensor = torch.tensor(all_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                    self.old_score = current_score

                self.game.latest_image = cv2.imread(os.path.join(game_folder, next))
                next_state, new_next_item = self.game.get_game_area()
                all_next_state = self.game.slot_with_item(next_state, new_next_item)
                next_score = self.game.get_score()
                next_state_tensor = torch.tensor(all_next_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

                if current_score == None:
                    current_score = 0
                elif next_score == None:
                    next_score = 0
                elif np.any(current_state >= 21):
                    print(current, "got 21")
                    current_state = None
                    continue
                elif np.any(next_state >= 21):
                    print(next, "got 21")
                    current_state = None
                    continue

                reward = self.get_reward(next_score)
                if torch.equal(current_state_tensor, next_state_tensor):
                    reward = torch.tensor([-1], device=self.device)
                elif next_action == 0 and current_action == 0:
                    reward = torch.tensor([-1], device=self.device)
                elif current_action == 0:
                    reward = torch.tensor([-0.1], device=self.device)
                reward_tensor = torch.tensor([reward], device=self.device)

                current_action_tensor = torch.tensor([current_action], device=self.device)
                # action = F.one_hot(current_action_tensor, num_classes=36).to(torch.int64)
                # print(action.shape)
                self.memory.push(current_state_tensor, current_action_tensor.unsqueeze(0), next_state_tensor, reward_tensor)

                current_state_tensor = next_state_tensor
                current_score = next_score
            else:
                current_state = None
            if current_num1 != next_num2:
                self.top_reward = 1
            
            print("length =",len(self.memory))
            if len(self.memory) >= load_size:
                break
        # return self.memory

    def select_action(self, all_state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * len(self.memory) / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # 1. calculate the probability distribution
                print("model")
                policy_output = self.policy_net(all_state)
                probabilities = F.softmax(policy_output.flatten(1), dim=1).view_as(policy_output)

        else:
            probabilities = torch.rand(36).to(self.device)
            print("random")

        # 2. create a mask for valid moves
        state_np = all_state.squeeze().cpu().numpy()
        state, next_item = self.game.split_result(state_np)
        valid_moves_mask = torch.flatten(torch.tensor(state)).to(self.device)
        if next_item == 17:
            for i in range(36):
                if valid_moves_mask[i] != 0:
                    valid_moves_mask[i] = 1
                else:
                    valid_moves_mask[i] = 0
        else:
            for i in range(36):
                if valid_moves_mask[i] == 0:
                    valid_moves_mask[i] = 1
                else:
                    valid_moves_mask[i] = 0
        valid_moves_mask[0] = 1  # the first position is always valid

        # filter out invalid moves
        probabilities *= valid_moves_mask
        probabilities /= probabilities.sum()

        # 3. select the best position
        action = torch.argmax(probabilities)
        # print("action:", action.item())
        # action_onehot = F.one_hot(action, num_classes=36).to(torch.int64)
        return action
    
    def get_reward(self, score):

        reward = score - self.old_score

        if reward > 30:
            time.sleep(1.5)
        elif reward > 100:
            time.sleep(4)

        if reward > self.top_reward:
            self.top_reward = reward

        if self.top_reward <= 0:
            self.top_reward = 1

        self.old_score = score
        return reward / self.top_reward

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_model(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self):
        torch.save(self.target_net.state_dict(), "target_net_parameters.pth")
        torch.save(self.policy_net.state_dict(), "policy_net_parameters.pth")
        torch.save(self.optimizer.state_dict(), "optimizer_parameters.pth")

    def load_model(self):
        policy_net_save = torch.load("policy_net_parameters.pth", weights_only=True)
        target_net_save = torch.load("target_net_parameters.pth", weights_only=True)
        optimizer_save = torch.load("optimizer_parameters.pth", weights_only=True)
        self.policy_net.load_state_dict(policy_net_save)
        self.target_net.load_state_dict(target_net_save)
        self.optimizer.load_state_dict(optimizer_save)

    def save_memory(self):
        with open("replay_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        with open("replay_memory.pkl", 'rb') as f:
            self.memory = pickle.load(f)