import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import pickle
import random
import torch

Transition = namedtuple('Transition', ('state', 'action', 'score', 'next_state', 'next_score'))

EnhancedTransition = namedtuple('EnhancedTransition', Transition._fields + ('train_reward',))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

ITEM_TYPE = 22
BROAD_SIZE = 6

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = Transition
        self.EnhancedTransition = EnhancedTransition
    
    def push(self, *args):
        transtion = Transition(*args)
        step_reward = transtion.next_score - transtion.score

        # if step_reward > 150:
        #     # train_reward = 0.3
        if step_reward > 600:
            train_reward = 0.5
        elif step_reward > 1000:
            train_reward = 0.7
        elif step_reward > 3000:
            train_reward = 0.9
        elif step_reward > 7500:
            train_reward = 1
        else:
            train_reward = 0

        state = transtion.state.squeeze()
        state_long = state.long()
        state_one_hot = F.one_hot(state_long, num_classes=ITEM_TYPE)
        state_one_hot = state_one_hot.permute(2, 0, 1)

        next_state = transtion.next_state.squeeze()
        next_state_long = next_state.long()
        next_state_one_hot = F.one_hot(next_state_long, num_classes=ITEM_TYPE)
        next_state_one_hot = next_state_one_hot.permute(2, 0, 1)

        if torch.equal(transtion.state, transtion.next_state):
            train_reward = -1
        train_reward_tensor = torch.tensor([train_reward], device=device)
        enhancedTransition = EnhancedTransition(state_one_hot.unsqueeze(0).float(),
                                                transtion.action,
                                                transtion.score,
                                                next_state_one_hot.unsqueeze(0).float(),
                                                transtion.next_score,
                                                train_reward_tensor)
        self.memory.append(enhancedTransition)
    
    def random_sample(self, batch_size):
        # batch = self.enhanced_sample()
        return random.sample(self.memory, batch_size)
    
    def sample(self):
        return self.memory
    
    # def enhanced_sample(self):
    #     batch = list(self.memory)
    #     enhanced_batch = []

    #     for i in range(len(batch)-1):            
    #         if batch[i].next_state is None:
    #             continue
    #         step_reward = batch[i].reward - batch[i+1].reward

    #         if step_reward > 150:
    #             train_reward = 0.3
    #         elif step_reward > 600:
    #             train_reward = 0.5
    #         elif step_reward > 1000:
    #             train_reward = 0.7
    #         elif step_reward > 3000:
    #             train_reward = 0.9
    #         elif step_reward > 7500:
    #             train_reward = 1
    #         else:
    #             train_reward = 0
            
    #         if torch.equal(batch[i].state, batch[i-1].state):
    #             train_reward = -1
    #         if batch[i].action.item() == 0 and torch.equal(batch[i].action, batch[i-1].action):
    #             train_reward = -1
            
    #         state = batch[i].state.squeeze()
    #         state_long = state.long()
    #         state_one_hot = F.one_hot(state_long, num_classes=ITEM_TYPE)
    #         state_one_hot = state_one_hot.permute(2, 0, 1)

    #         next_state = batch[i].next_state.squeeze()
    #         next_state_long = next_state.long()
    #         next_state_one_hot = F.one_hot(next_state_long, num_classes=ITEM_TYPE)
    #         next_state_one_hot = next_state_one_hot.permute(2, 0, 1)

    #         train_reward_tensor = torch.tensor([train_reward], device=device)
    #         enhanced_transition = EnhancedTransition(state_one_hot.unsqueeze(0).float(),
    #                                                  batch[i].action,
    #                                                  next_state_one_hot.unsqueeze(0).float(),
    #                                                  batch[i].reward,
    #                                                  train_reward_tensor)
    #         enhanced_batch.append(enhanced_transition)
    #     return enhanced_batch
    
    def save_memory(self):
        with open("replay_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        with open("replay_memory.pkl", 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, item_type = ITEM_TYPE, board_size = BROAD_SIZE):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(item_type, 128, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(144, 128)
        self.fc2 = nn.Linear(128, board_size * board_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x