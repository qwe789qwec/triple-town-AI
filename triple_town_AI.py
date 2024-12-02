import math
import numpy as np
import random
import time
import cv2
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import triple_town_model
from triple_town_game import playgame

ITEM_TYPE = 22
BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
BATCH_SIZE = 300
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class TripleTownAI:
    def __init__(self, 
                 item_type = ITEM_TYPE,
                 broad_size=BROAD_SIZE, 
                 batch_size=BATCH_SIZE, 
                 gamma=GAMMA, 
                 eps_start=EPS_START, 
                 eps_end=EPS_END, 
                 eps_decay=EPS_DECAY, 
                 tau = TAU, 
                 learning_rate = LR, 
                 memory_size=10000):
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
        self.Transition = self.memory.EnhancedTransition
        self.game = playgame()

        self.policy_net = triple_town_model.DQN(item_type ,broad_size).to(self.device)
        self.target_net = triple_town_model.DQN(item_type ,broad_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

    def select_action(self, all_state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * len(self.memory) / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # 1. calculate the probability distribution
                print("model")
                state_long = all_state.squeeze().long()
                state_one_hot = F.one_hot(state_long, num_classes=ITEM_TYPE)
                state_one_hot = state_one_hot.permute(2, 0, 1)
                policy_output = self.policy_net(state_one_hot.unsqueeze(0).float())
                print(policy_output.shape)
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

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.random_sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.train_reward)

        # print("reward_batch:", reward_batch.shape)

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

    def get_file_info(self, file_name):
        part_before_game = file_name.split("_info_")[0]
        part_after_game = file_name.split("_info_")[1]

        game_info = part_before_game.replace("game_", "").split("_")
        num, step, action = map(int, game_info)

        split_str = part_after_game.replace(".png", "").split('_')
        next_item = int(split_str[0])
        score = int(split_str[1])
        matrix_elements = list(map(int, split_str[2:]))
        state = np.array(matrix_elements).reshape(6, 6)

        return num, step, action, next_item, score, state

    def load_memory(self, load_size=150, skip=0):
        game_folder = 'gameplay'
        print("start load memory")
        image_files = sorted(
            [f for f in os.listdir(game_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda f: os.path.getmtime(os.path.join(game_folder, f))
        )
        
        for i in range(len(image_files)-1):
            j = i + 1
            current = image_files[i]
            next = image_files[j]

            if "_info_" in current and "_info_" in next:
                print(current)
                current_num, current_step, current_action, current_next_item, current_score, current_state = self.get_file_info(current)
                next_num, next_step, next_action, next_item, next_score, next_state = self.get_file_info(next)

                if current_num < skip:
                    continue
                if current_num == next_num and next_step - current_step == 1:
                    all_state = self.game.slot_with_item(current_state, current_next_item)
                    current_state_tensor = torch.tensor(all_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

                    all_next_state = self.game.slot_with_item(next_state, next_item)
                    next_state_tensor = torch.tensor(all_next_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

                    if current_score == None:
                        current_score = 0
                    elif next_score == None:
                        next_score = 0
                    if np.any(current_state >= 21):
                        print(current, "got 21")
                        current_state = None
                        continue
                    if np.any(next_state >= 21):
                        print(next, "got 21")
                        next_state = None
                        continue

                    reward = next_score
                    reward_tensor = torch.tensor([reward], device=self.device)

                    current_action_tensor = torch.tensor([current_action], device=self.device)
                    self.memory.push(current_state_tensor, current_action_tensor.unsqueeze(0), next_state_tensor, reward_tensor)

            if len(self.memory) >= load_size:
                break
        
        print("memory length:", len(self.memory.sample()))

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