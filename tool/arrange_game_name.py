import math
import numpy as np
import random
import time
import cv2
import os
import sys
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
    
    part_before_info = image_file.split("_info_")[0]
    part_before_info = part_before_info.replace("game_", "")
    game_info = part_before_info.split("_")
    num, step, action = map(int, game_info)
    action_str = str(action) + ".png"
    num = num
    step = step
    new_name = "_".join(["game", str(num), str(step), action_str])

    part_after_info = image_file.split("_info_")[1]
    part_after_info = part_after_info.replace(".png", "")

    # new_name = "game_" + part_before_info + ".png"
    print(new_name)
    new_path = os.path.join(game_folder, new_name)
    image_path = os.path.join(game_folder, image_file)
    os.rename(image_path, new_path)
    # break