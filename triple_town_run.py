import time
from itertools import count

import torch
import torch.nn.functional as F

import triple_town_model
from triple_town_AI import TripleTownAI
from triple_town_game import playgame

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BROAD_SIZE = 6
ACTION_SPACE = BROAD_SIZE * BROAD_SIZE
ITEM_TYPE = 22
BATCH_SIZE = 150
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000
LOAD_SIZE = 10000
SKIP_GAME = 0

game = playgame()

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

tpai.load_model()
# tpai.memory.load_memory()
# tpai.optimize_model()
tpai.load_memory(LOAD_SIZE, SKIP_GAME)
# tpai.memory.save_memory()

# if len(tpai.memory.sample()) > BATCH_SIZE:
#     for i in range(len(tpai.memory.sample())-BATCH_SIZE):
#         tpai.optimize_model()
#         tpai.update_model()
# print("finish optimize model and update model")

tpai.save_model()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 2
    torch.cuda.empty_cache()
else:
    num_episodes = 3

# memory = load_memory_json()
# print("memory length:", len(memory))

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    game.take_screenshot()
    state, next_item = game.get_game_area()
    all_state = game.slot_with_item(state, next_item)
    state_tensor = torch.tensor(all_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    top_reward = 1
    score = game.get_score()
    if score == None:
        game.take_screenshot()
        score = game.get_score()
        if score == None:
            score = 0
    old_pos_number = 0

    for t in count():

        action = tpai.select_action(state_tensor)
        game.click_slot(action.item())
        game.take_screenshot()
        new_score = game.get_score()
        if new_score == None:
            time.sleep(1)
            game.take_screenshot()
            new_score = game.get_score()
            if new_score == None:
                new_score = 0
        reward = new_score

        print("action:", action.item())
        print("next_item:", next_item)
        print("score:", new_score)
        print("game_step:", t)

        if(game.is_game_end()):
            next_state_tensor = None
            game.game_number = game.get_next_game_number()
            game.step = 0
            game.restart_game()
        else:
            observation, new_next_item = game.get_game_area()
            all_observation = game.slot_with_item(observation, new_next_item)
            next_state_tensor = torch.tensor(all_observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
        old_pos_number = action.item()
        reward = new_score
        reward_tensor = torch.tensor([reward], device=device)

        tpai.memory.push(state_tensor, action.unsqueeze(0).unsqueeze(0), next_state_tensor, reward_tensor)

        state = observation
        next_item = new_next_item
        state_tensor = next_state_tensor
        print("state:\n", state)
        print("reward:", reward_tensor.item())
        print("=========================================================")

        # Perform one step of the optimization (on the policy network)
        tpai.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tpai.update_model()

        if next_state_tensor is None:
            break
    tpai.save_model()
    # tpai.memory.save_memory()

print('Complete')