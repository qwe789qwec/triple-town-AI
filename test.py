import cv2
import os
from playgame import playgame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(10000)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

game = playgame()

def load_memory():
    game_folder = 'gameplay'
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
                continue
            elif np.any(next_state >= 21):
                print(next, "got 21")
                continue

            reward = next_score - current_score
            if torch.equal(current_state_tensor, next_state_tensor):
                reward = torch.tensor([-1000], device=device)
            elif next_action == 0 and current_action == 0:
                reward = torch.tensor([-1000], device=device)
            reward_tensor = torch.tensor([reward], device=device)

            current_action_tensor = torch.tensor([current_action], device=device)
            action = F.one_hot(current_action_tensor, num_classes=36).to(torch.int64)
            memory_next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            memory.push(current_state_tensor, action.unsqueeze(0), memory_next_state, reward_tensor)

            current_state_tensor = next_state_tensor
            current_score = next_score
        else:
            current_state = None