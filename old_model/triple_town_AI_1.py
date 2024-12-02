import time
from triple_town_game import playgame
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
game = playgame()
mouse_init_x = game.screen_x + 70
mouse_init_y = game.screen_y + 160

def mouse_click(pos):
    x = pos % 6
    y = pos // 6
    pyautogui.moveTo(mouse_init_x + slot_gap * x, mouse_init_y + slot_gap * y)
    pyautogui.click()
    if pos < 12:
        time.sleep(1.0)
    time.sleep(0.5)

def restart_game():
    pyautogui.click(game.end_x, game.end_y)
    time.sleep(3)
    pyautogui.click(game.start_x, game.start_y)
    time.sleep(3)

# while True:
#     if(gamesc.is_game_end()):
#         restart_game()
#         break
#     gamesc.take_screenshot()
#     # gamesc.show_image(gamesc.latest_image)
#     gamesc.save_image(gamesc.latest_image)
#     random_number = np.random.randint(1, 37)
#     mouse_click(random_number)
#     # game_area = gamesc.get_game_area()
#     # gamesc.save_image(game_area)
#     print(gamesc.get_score())
#     time.sleep(1)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

STATE_SHAPE = (3, 512, 512)
ACTION_SPACE = 36
BATCH_SIZE = 10
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

# define DQN model
class DQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128* 60 * 60, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output of the conv layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

policy_net = DQN(STATE_SHAPE, ACTION_SPACE).to(device)
target_net = DQN(STATE_SHAPE, ACTION_SPACE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)
    else:
        action_index = torch.randint(0, 36, (1,), device=device)
        action_onehot = F.one_hot(action_index, num_classes=36).to(torch.float32)
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
    action_batch = torch.cat(batch.action).unsqueeze(0)
    # action_batch = action_batch.long()
    print("PP:", policy_net(state_batch).shape,"aa:", action_batch.shape)
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
    num_episodes = 600
    torch.cuda.empty_cache()
else:
    num_episodes = 50
    
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    game.take_screenshot()
    state = game.get_game_area()
    score = game.get_score()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state = state.permute(0, 3, 1, 2)
    # print(state.shape)

    action_onehot = select_action(state)
    action = action_onehot.max(1).indices
    action_number = action.item()
    game.save_image(game.latest_image, action_number)
    mouse_click(action_number)
    observation = game.get_game_area()
    new_score = game.get_score()
    reward = new_score - score
    if(game.is_game_end()):
        terminated = True
        restart_game()
    else:
        terminated = False
    reward = torch.tensor([reward], device=device)

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = next_state.permute(0, 3, 1, 2)

    memory.push(state, action, next_state, reward)
    state = next_state

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