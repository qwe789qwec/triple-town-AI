import time
from itertools import count

import torch

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
ITEM_SPACE = 25
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000
LOAD_SIZE = 150

game = playgame()

# policy_net_save = torch.load("policy_net_parameters.pth", weights_only=True)
# target_net_save = torch.load("target_net_parameters.pth", weights_only=True)
# optimizer_save = torch.load("optimizer_parameters.pth", weights_only=True)
# policy_net.load_state_dict(policy_net_save)
# target_net.load_state_dict(target_net_save)
# optimizer.load_state_dict(optimizer_save)

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

tpai.load_memory(LOAD_SIZE)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10
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
    tpai.old_reward = score
    old_pos_number = 0

    for t in count():

        action = tpai.select_action(state_tensor)
        game.mouse_click(action.item())
        game.take_screenshot()
        new_score = game.get_score()
        if new_score == None:
            time.sleep(1)
            game.take_screenshot()
            new_score = game.get_score()
            if new_score == None:
                score = 0

        if(game.is_game_end()):
            next_state_tensor = None
            game.game_number = game.get_next_game_number()
            game.step = 0
            game.restart_game()
        else:
            observation, new_next_item = game.get_game_area()
            all_observation = game.slot_with_item(observation, new_next_item)
            next_state_tensor = torch.tensor(all_observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            reward = tpai.get_reward(new_score)
            # pos = action.max(0).indices
            # pos_number = pos.item()
            if torch.equal(state_tensor, next_state_tensor):
                reward = torch.tensor([-1], device=device)
            elif old_pos_number == action.item():
                reward = torch.tensor([-1], device=device)
            elif action.item() == 0:
                reward = torch.tensor([-0.1], device=device)

        old_pos_number = action.item()
        reward_tensor = torch.tensor([reward], device=device)

        tpai.memory.push(state_tensor, action.unsqueeze(0).unsqueeze(0), next_state_tensor, reward_tensor)
        print("reward:", reward_tensor.item())
        print("=========================================================")

        state = observation
        next_item = new_next_item
        state_tensor = next_state_tensor

        print("state:\n", state)
        print("next_item:", next_item)
        print("score:", score)
        print("game_step:", t)

        # Perform one step of the optimization (on the policy network)
        tpai.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        tpai.update_model()
    tpai.save_model()

print('Complete')