import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = Transition
    
    def push(self, *args):
        self.memory.append(self.Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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