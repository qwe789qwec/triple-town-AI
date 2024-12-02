import math
import numpy as np
import random
import time
import cv2
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from triple_town_game import playgame

game_folder = 'gameplay'
image_files = sorted(
    [f for f in os.listdir(game_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda f: os.path.getmtime(os.path.join(game_folder, f))
)
current_state = None

game = playgame()

for image_file in image_files:
    if image_file.startswith("game_9_"):
        print(image_file)
    else:
        continue
    
    game_info = image_file.replace("game_", "").replace(".png", "")
    game_info = game_info.split("_")
    num, step, action = map(int, game_info)
    action_str = str(action) + ".png"
    num = num
    step = step
    new_name = "_".join(["game", str(num), str(step), action_str])

    part_before_info = image_file.split("_info_")[0]
    new_name = part_before_info + ".png"
    print(new_name)
    new_path = os.path.join(game_folder, new_name)
    image_path = os.path.join(game_folder, image_file)
    # os.rename(image_path, new_path)
    break