from triple_town_AI import TripleTownAI
from collections import namedtuple, deque
import torch
import torch.nn.functional as F

BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
ITEM_TYPE = 30
BATCH_SIZE = 10
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000
LOAD_SIZE = 20
SKIP_GAME = 0

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
EnhancedTransition = namedtuple('EnhancedTransition', Transition._fields + ('train_reward',))

tpai = TripleTownAI(
    item_type=ITEM_TYPE,
    broad_size=BROAD_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    eps_start=EPS_START,
    eps_end=EPS_END,
    eps_decay=EPS_DECAY,
    tau=TAU,
    learning_rate=LR,
    memory_size=MEMORY_SIZE
)

tpai.load_new_memory()

batch = list(tpai.memory.sample())
num_classes = 30

for i in reversed(range(len(batch))):
    # item = batch[i].state[0, 0, 3, 3].item()
    step_reward = batch[i].reward - batch[i-1].reward
    
    print(batch[i].state.shape)  # (1, 3, 6, 6)
    state = batch[i].state.squeeze()
    state_long = state.long()
    one_hot = F.one_hot(state_long, num_classes=num_classes)
    one_hot = one_hot.permute(2, 0, 1)

    print(one_hot.shape)  # (14, 6, 6)