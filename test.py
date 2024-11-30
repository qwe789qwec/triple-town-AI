from triple_town_AI import TripleTownAI
from collections import namedtuple, deque

BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
ITEM_SPACE = 25
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

tpai.memory.load_memory()

batch = tpai.memory.sample()
print("batch length:", len(batch))