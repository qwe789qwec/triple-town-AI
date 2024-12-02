import math
import numpy as np
import random
import time
import cv2
import os
from triple_town_game import playgame

game_folder = 'gameplay'
image_files = sorted(
    [f for f in os.listdir(game_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda f: os.path.getmtime(os.path.join(game_folder, f))
)
current_state = None

game = playgame()

for image_file in image_files:
    if "_info_" in image_file:
        print(image_file)
    else:
        image_path = os.path.join(game_folder, image_file)
        game.latest_image = cv2.imread(image_path)
        state, next_item = game.get_game_area()
        score = game.get_score()
        if score == None:
            score = 0

        arr_str = "_".join(map(str, state.flatten()))
        arr_str = arr_str + ".png"
        game_info = image_file.replace(".png", "")
        new_name = "_".join([game_info, "info", str(next_item), str(score), arr_str])

        print(new_name)
        new_path = os.path.join(game_folder, new_name)
        os.rename(image_path, new_path)
    # break

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