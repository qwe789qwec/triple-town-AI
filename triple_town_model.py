import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import pickle
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
EnhancedTransition = namedtuple('EnhancedTransition', Transition._fields + ('train_reward',))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = Transition
        self.EnhancedTransition = EnhancedTransition
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def random_sample(self, batch_size):
        batch = self.enhanced_sample()
        return random.sample(batch, batch_size)
    
    def sample(self):
        return self.memory
    
    def enhanced_sample(self):
        batch = list(self.memory)
        enhanced_batch = []

        old_gap = 100
        factor = 1

        for i in reversed(range(len(batch))):
            if i == 0:
                break
            
            gap = batch[i].reward - batch[i-1].reward

            if gap > old_gap:
                old_gap = gap
                factor = 1

            train_reward = gap * factor
            factor *= 0.5

            enhanced_transition = EnhancedTransition(*batch[i], train_reward)
            enhanced_batch.append(enhanced_transition)
        return enhanced_batch
    
    def save_memory(self):
        with open("replay_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        with open("replay_memory.pkl", 'rb') as f:
            self.memory = pickle.load(f)

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