import time
from playgame import playgame
import pyautogui
import math
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

slot_gap = 80
gamesc = playgame()
mouse_init_x = gamesc.screen_x + 70
mouse_init_y = gamesc.screen_y + 160
end_x = gamesc.screen_x + 682
end_y = gamesc.screen_y + 247
start_x = gamesc.screen_x + 97
start_y = gamesc.screen_y + 198

def mouse_click(pos_onehot):
    # pos = pos + 1
    # x = pos % 6
    # y = pos // 6

    pos = pos_onehot.max(0).indices
    pos_number = pos.item()
    gamesc.save_image(gamesc.latest_image, pos_number)

    row, col = divmod(pos_number, 6)
    pyautogui.moveTo(mouse_init_x + slot_gap * col, mouse_init_y + slot_gap * row)
    pyautogui.click()
    if pos_number in {2, 3}:
        time.sleep(5.0)
    else:
        time.sleep(1.0)


def restart_game():
    pyautogui.click(end_x, end_y)
    time.sleep(3)
    pyautogui.click(start_x, start_y)
    time.sleep(3)

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
                        ('state', 'item', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# define DQN model
class DQN(nn.Module):
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, board_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(board_size, board_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(board_size, board_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(board_size * board_size * board_size + 1 , 128)
        self.fc2 = nn.Linear(128, board_size * board_size)

    def forward(self, x, next_element):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        next_element = next_element / 25
        x = torch.cat([x, next_element], dim=1)
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

steps_done = 0
def select_action(state, next_item):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # 1. calculate the probability distribution
            print("model")
            policy_output = policy_net(state, next_item)
            probabilities = F.softmax(policy_output.flatten(1), dim=1).view_as(policy_output)

    else:
        probabilities = torch.rand(36).to(device)
        print("random")

    # 2. create a mask for valid moves
    valid_moves_mask = torch.flatten(state, start_dim=1).clone().to(device).squeeze(0)
    if next_item.item() == 17:
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
    non_final_next_states = torch.cat([s for s, i in zip(batch.next_state, batch.item) 
                                       if s is not None])
    non_final_next_items = torch.cat([i for s, i in zip(batch.next_state, batch.item) 
                                      if s is not None])


    state_batch = torch.cat(batch.state)
    next_item_batch = torch.cat(batch.item)
    action_batch = torch.cat(batch.action)
    # print("state_batch:", state_batch.shape)
    # print("next_item_batch:", next_item_batch.shape)
    # print("action_batch:", action_batch.shape)
    # action_batch = action_batch.long()
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch, next_item_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states, non_final_next_items).max(1).values
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

gamesc.take_screenshot()
state, next_item = gamesc.get_game_area()
state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
next_item_tensor = torch.tensor([next_item], dtype=torch.float32, device=device).unsqueeze(0)
first_run_flag = False
old_pos_number = 0

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    # gamesc.take_screenshot()
    # state, next_item = gamesc.get_game_area()
    print("state:\n", state_tensor)
    print("next_item:", next_item_tensor.item())
    # state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    # next_item_tensor = torch.tensor([next_item], dtype=torch.float32, device=device).unsqueeze(0)
    score = gamesc.get_score()
    if first_run_flag:
        score = 0
        first_run_flag = False
    # print("score:", score)
    
    action = select_action(state_tensor, next_item_tensor)
    mouse_click(action)
    print("episode:", i_episode)

    gamesc.take_screenshot()
    if(gamesc.is_game_end()):
        next_state_tensor = None
        # next_item_tensor = None
        gamesc.game_number = gamesc.get_next_game_number()
        new_score = gamesc.get_score() * 10
        gamesc.step = 0
        restart_game()
        reward_tensor = torch.tensor([new_score], device=device)
        time.sleep(1)
    else:
        observation, new_item = gamesc.get_game_area()
        new_score = gamesc.get_score()
        reward = (new_score - score) * 10
        # print("old_score:", score)
        print("new_score:", new_score)
        if reward > 30:
            time.sleep(1.0)
        elif reward > 100:
            time.sleep(3)

        next_state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        new_item_tensor = torch.tensor([new_item], dtype=torch.float32, device=device).unsqueeze(0)
        
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
    
    memory.push(state_tensor, next_item_tensor, action.unsqueeze(0), next_state_tensor, reward_tensor)
    print("reward:", reward_tensor.item())
    print("=========================================================")

    if next_state_tensor is None:
        first_run_flag = True
        gamesc.take_screenshot()
        observation, new_item = gamesc.get_game_area()
        next_state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        new_item_tensor = torch.tensor([new_item], dtype=torch.float32, device=device).unsqueeze(0)
    state_tensor = next_state_tensor
    next_item_tensor = new_item_tensor

    if i_episode % 150 == 0:
        torch.save(target_net.state_dict(), "target_net_parameters.pth")
        torch.save(policy_net.state_dict(), "policy_net_parameters.pth")
        torch.save(optimizer.state_dict(), "optimizer_parameters.pth")

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