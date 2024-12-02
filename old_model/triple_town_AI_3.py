import time
from triple_town_game import playgame
import pyautogui
import math
import numpy as np
import random
from collections import namedtuple, deque

import cv2
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

slot_gap = 80
game = playgame()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
ITEM_SPACE = 25
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# def save_memory_json(memory, filepath='replay_memory.json'):
#     data = [transition._asdict() for transition in memory.memory]
#     with open(filepath, 'w') as f:
#         json.dump(data, f)

# def load_memory_json(filepath='replay_memory.json'):
#     with open(filepath, 'r') as f:
#         data = json.load(f)
#     loaded_memory = ReplayMemory(len(data))
#     for item in data:
#         loaded_memory.push(
#             torch.tensor(item['state']), 
#             torch.tensor(item['action']), 
#             torch.tensor(item['next_state']), 
#             torch.tensor(item['reward'])
#         )
#     return loaded_memory

# define DQN model
class DQN(nn.Module):
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, board_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(board_size, board_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(board_size, board_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((board_size-1) * (board_size-1) * board_size, 128)
        self.fc2 = nn.Linear(128, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x

policy_net = DQN(BROAD_SIZE).to(device)
target_net = DQN(BROAD_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# policy_net_save = torch.load("policy_net_parameters.pth", weights_only=True)
# target_net_save = torch.load("target_net_parameters.pth", weights_only=True)
# optimizer_save = torch.load("optimizer_parameters.pth", weights_only=True)
# policy_net.load_state_dict(policy_net_save)
# target_net.load_state_dict(target_net_save)
# optimizer.load_state_dict(optimizer_save)
memory = ReplayMemory(10000)

def load_memory(load_size=150):
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

        if current_num1 == next_num2 and next_step - current_step == 1:
            if current_state is None:
                game.latest_image = cv2.imread(os.path.join(game_folder, current))
                current_state, next_item = game.get_game_area()
                all_state = game.slot_with_item(current_state, next_item)
                current_score = game.get_score()
                current_state_tensor = torch.tensor(all_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            game.latest_image = cv2.imread(os.path.join(game_folder, next))
            next_state, new_next_item = game.get_game_area()
            all_next_state = game.slot_with_item(next_state, new_next_item)
            next_score = game.get_score()
            next_state_tensor = torch.tensor(all_next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

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

            reward = (next_score - current_score) * 10
            if torch.equal(current_state_tensor, next_state_tensor):
                reward = torch.tensor([-1000], device=device)
            elif next_action == 0 and current_action == 0:
                reward = torch.tensor([-1000], device=device)
            reward_tensor = torch.tensor([reward], device=device)

            current_action_tensor = torch.tensor([current_action], device=device)
            action = F.one_hot(current_action_tensor, num_classes=36).to(torch.int64)
            # print(action.shape)
            memory.push(current_state_tensor, action, next_state_tensor, reward_tensor)

            current_state_tensor = next_state_tensor
            current_score = next_score
        else:
            current_state = None
        
        print("length =",len(memory))
        if len(memory) >= load_size:
            break

def select_action(all_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * len(memory) / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            # 1. calculate the probability distribution
            print("model")
            policy_output = policy_net(all_state)
            probabilities = F.softmax(policy_output.flatten(1), dim=1).view_as(policy_output)

    else:
        probabilities = torch.rand(36).to(device)
        print("random")

    # 2. create a mask for valid moves
    state_np = all_state.squeeze().cpu().numpy()
    state, next_item = game.split_result(state_np)
    valid_moves_mask = torch.flatten(torch.tensor(state)).to(device)
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
    print("action:", action.item())
    action_onehot = F.one_hot(action, num_classes=36).to(torch.int64)
    return action_onehot

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10000
    torch.cuda.empty_cache()
else:
    num_episodes = 50

# load_memory(1000)
# memory = load_memory_json()
# print("memory length:", len(memory))
game.take_screenshot()
state, next_item = game.get_game_area()
all_state = game.slot_with_item(state, next_item)
state_tensor = torch.tensor(all_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
old_pos_number = 0

for i_episode in range(num_episodes):
    # Initialize the environment and get its state

    score = game.get_score()
    if score == None:
        game.take_screenshot()
        score = game.get_score()
        if score == None:
            score = 0

    state_np = state_tensor.squeeze().cpu().numpy()
    state ,next_item = game.split_result(state_np)
    print("state:\n", state)
    print("next_item:", next_item)
    print("score:", score)
    print("episode:", i_episode)

    action = select_action(state_tensor)
    game.mouse_click(action)
    game.take_screenshot()
    new_score = game.get_score()

    if new_score == None:
        time.sleep(1)
        game.take_screenshot()
        new_score = game.get_score()
        if new_score == None:
            score = 0
    reward = (new_score - score) * 10
    
    if reward > 30:
        time.sleep(1)
    elif reward > 100:
        time.sleep(3)

    if(game.is_game_end()):
        next_state_tensor = None
        game.game_number = game.get_next_game_number()
        new_score = game.get_score() * 10
        if new_score == None:
            new_score = game.last_number
        game.step = 0
        game.restart_game()
        reward_tensor = torch.tensor([new_score], device=device)
    else:
        observation, new_next_item = game.get_game_area()
        all_observation = game.slot_with_item(observation, new_next_item)

        next_state_tensor = torch.tensor(all_observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        pos = action.max(0).indices
        pos_number = pos.item()
        if torch.equal(state_tensor, next_state_tensor):
            reward = torch.tensor([-1000], device=device)
        elif old_pos_number == pos_number:
            reward = torch.tensor([-1000], device=device)
        elif pos_number == 0:
            reward = torch.tensor([-10], device=device)

        old_pos_number = pos_number
        reward_tensor = torch.tensor([reward], device=device)
    
    print("action shape:",action.shape)
    memory.push(state_tensor, action.unsqueeze(0), next_state_tensor, reward_tensor)
    print("reward:", reward_tensor.item())
    print("=========================================================")

    state_tensor = next_state_tensor
    if next_state_tensor is None:
        game.take_screenshot()
        state, next_item = game.get_game_area()
        all_state = game.slot_with_item(state, next_item)
        state_tensor = torch.tensor(all_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    if i_episode % 150 == 0:
        torch.save(target_net.state_dict(), "target_net_parameters.pth")
        torch.save(policy_net.state_dict(), "policy_net_parameters.pth")
        torch.save(optimizer.state_dict(), "optimizer_parameters.pth")
        # save_memory_json(memory)

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

print('Complete')