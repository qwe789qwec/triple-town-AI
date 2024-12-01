import math
import numpy as np
import random
import time
import cv2
import os
from triple_town_game import playgame

game_folder = 'arrange'
image_files = sorted(
    [f for f in os.listdir(game_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda f: os.path.getmtime(os.path.join(game_folder, f))
)
current_state = None

game = playgame()

for image_file in image_files:
    if image_file.startswith("game_"):
        print(image_file)
    else:
        continue

    game_info = image_file.replace("game_", "").replace(".png", "")
    game_info = game_info.split("_")
    num, step, action = map(int, game_info)
    action_str = str(action) + ".png"
    num = num + 34
    new_name = "_".join(["game", str(num), str(step), action_str])

    print(new_name)
    new_path = os.path.join(game_folder, new_name)
    image_path = os.path.join(game_folder, image_file)
    os.rename(image_path, new_path)
#     break

# part_before_game = new_name.split("_info_")[0]
# part_after_game = new_name.split("_info_")[1]

# game_info = part_before_game.split("_")
# current_num1, current_step, current_action = map(int, game_info)

# split_str = part_after_game.replace(".png", "").split('_')
# next_item = int(split_str[0])
# score = int(split_str[1])
# matrix_elements = split_str[2:]
# state = np.array(matrix_elements).reshape(6, 6)

# print("game number:", current_num1)
# print("game step:", current_step)
# print("game action:", current_action)
# print("next item:", next_item)
# print("score:", score)
# print("state:\n", state)