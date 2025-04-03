import pyautogui
import time
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from PIL import Image
import easyocr
from pathlib import Path
from collections import namedtuple
from triple_town_simulate import TripleTownSim

position = namedtuple('position', ['x', 'y'])
region = namedtuple('region', ['x', 'y', 'w', 'h'])
size = namedtuple('size', ['w', 'h'])

class TripleTownHandler:
    def __init__(self, screen_dir='gameplay', output_dir='output'):
        self.step = 0
        self.screen_dir = screen_dir
        os.makedirs(self.screen_dir, exist_ok=True)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.screen_center = self._get_item_position()
        self.game_region = region(self.screen_center.x - 400, self.screen_center.y - 15, 800, 630)

        # mouse standby and slot init 2522 251
        self.standby = position(self.game_region.x + 530, self.game_region.y + 100)
        pyautogui.moveTo(self.standby.x, self.standby.y)
        pyautogui.click()
        # 2065 309
        self.slot_init = position(self.game_region.x + 70, self.game_region.y + 160)

        self.slot_capture = position(70, 160)
        self.score = region(163, 41, 210, 40)
        self.slot_gap = 80
        self.slot_size = 60

        self.game_number = self._get_next_game_number()
        self.score = 0
        self.last_score = 0

    def _take_screenshot(self, region=None):
        if region is not None:
            pyautogui.moveTo(self.standby.x, self.standby.y)
            pyautogui.click()
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.game_shot = frame
        return frame

    def _get_item_position(self, image_path='buttons/window.png'):
        screenshot = self._take_screenshot()
        template = cv2.imread(image_path, cv2.IMREAD_COLOR)
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # get window position
        if max_val > 0.8:  # matching threshold
            center_x = max_loc[0] + template.shape[1] // 2
            center_y = max_loc[1] + template.shape[0] // 2
            print(f"window position: {center_x}, {center_y}")
            return position(center_x, center_y)
        else:
            print("no matching window found!")
            return position(-1, -1)

    def _get_next_game_number(self):
        os.makedirs(self.screen_dir, exist_ok=True)
        existing_files = os.listdir(self.screen_dir)

        game_numbers = []
        for filename in existing_files:
            if filename.startswith("game_") and filename.endswith(".png"):
                try:
                    # get the game number from the filename
                    game_number = int(filename.split("_")[1])
                    game_numbers.append(game_number)
                except ValueError:
                    pass  # ignore invalid filenames

        return max(game_numbers, default=0) + 1

    def save_image(self, image, action=-1):
        if image is None:
            print("No image to save.")
            return
        gameinfo = "_".join(map(str, self.status.flatten()))
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.step = self.step + 1
        filename = f"game_{self.game_number}_{self.step}_{self.score}_{action}_info_{gameinfo}.png"
        pil_image.save(os.path.join(self.screen_dir, filename))

    def _get_score(self, take_screenshot=True):
        score_str = None
        score = 0
        check_time = 0
        reader = easyocr.Reader(['en'], gpu = True)

        while score is None and check_time < 2:
            score_region = self.game_shot[self.score.y:self.score.y + self.score.h, self.score.x:self.score.x + self.score.w]
            score_str = reader.readtext(cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY))
            check_time += 1
            if score_str is None and check_time < 2 and take_screenshot:
                time.sleep(1)
                self._take_screenshot(region=self.game_region)
        
        if score_str == [] or score_str is None:
            return 0
        else:
            try:
                if score_str[0][1].count(',') > 0:
                    score = int(score_str[0][1].replace(',', ''))
                else:
                    score = int(score_str[0][1])
                self.last_score = score
            except ValueError:
                return self.last_score
        self.score = score
        return self.score
    
    def game_status(self, take_screenshot=True):
        self._take_screenshot(region=self.game_region)
        max_retries = 3
        status = np.zeros(37, dtype=int)

        for slot_num in range(37):
            retry_count = 0
            if slot_num == 0:
                check_pos = position(500, 70)
            else:
                row, col = divmod(slot_num - 1, 6)
                slot_x = self.slot_capture.x + self.slot_gap * col - (self.slot_size // 2)
                slot_y = self.slot_capture.y + self.slot_gap * row - (self.slot_size // 2)
                check_pos = position(slot_x, slot_y)

            
            while retry_count < max_retries:
                slot_img = self.game_shot[check_pos.y:check_pos.y + self.slot_size, check_pos.x:check_pos.x + self.slot_size]
                index = self._find_matching_item(slot_img)
                # cv2.imwrite(f"test/slot_{slot_num}.png", slot_img)
                if index >= 21:
                    retry_count += 1
                    status[slot_num] = 21
                    if take_screenshot and retry_count < max_retries:
                        print(f"Failed to recognize slot {slot_num}, taking screenshot.")
                        time.sleep(1.5)
                        self._take_screenshot(region=self.game_region)
                else:
                    status[slot_num] = index
                    break

        self.status = status
        return status

    def _find_matching_item(self, item_image):
        max_match_value = 0.6
        matching_item_id = None
        item_image_gray = cv2.cvtColor(item_image, cv2.COLOR_BGR2GRAY)

        # compare the item image with the template images
        for item_folder in os.listdir("item_template"):
            item_folder_path = os.path.join("item_template", item_folder)
            
            if os.path.isdir(item_folder_path):
                for template_file in os.listdir(item_folder_path):
                    template_image_path = os.path.join(item_folder_path, template_file)
                    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
                    if template_image is None:
                        continue
                    score, _ = compare_ssim(item_image_gray, template_image, full=True)
                    
                    if score > max_match_value:
                        max_match_value = score
                        matching_item_id = item_folder
                    if score > 0.8 and matching_item_id is not None:
                        if not template_file.startswith("0_"):
                            new_name = f"0_{template_file}"
                            new_path = os.path.join(item_folder_path, new_name)
                            os.rename(template_image_path, new_path)
                            print(f"Renamed: {template_image_path} -> {new_path}")
                        break

        # if no matching item found, create a new folder
        # else, save the new item image
        if matching_item_id is None:
            print("No match found, creating a new folder for this item.")
            new_item_id = str(self._get_next_path_id("item_template"))
            new_folder = os.path.join("item_template", new_item_id)
            new_folder = Path(new_folder)
            new_folder.mkdir(parents=True, exist_ok=True)
            image_index = self._get_next_path_id(new_folder)
            cv2.imwrite(os.path.join("item_template", new_item_id, f"{image_index}.png"), item_image)
            matching_item_id = new_item_id
        else:
            image_index = self._get_next_path_id(os.path.join("item_template", matching_item_id))
            if image_index < 500 and max_match_value < 0.8:
                cv2.imwrite(os.path.join("item_template", matching_item_id, f"{image_index}.png"), item_image)

        return int(matching_item_id)
    
    def stadnby(self):
        pyautogui.moveTo(self.standby.x, self.standby.y)
        pyautogui.click()
        time.sleep(1.0)

    def click_slot(self, action):
        # play game
        row, col = divmod(action, 6)
        pyautogui.moveTo(self.slot_init.x + self.slot_gap * col, self.slot_init.y + self.slot_gap * row)
        pyautogui.click()
        if action in {2, 3}:
            time.sleep(5.0)
        else:
            time.sleep(1.0)
    
    def _get_next_path_id(self, folder_path):
        existing_ids = []
        for item_folder in os.listdir(folder_path):
            folder_name, _ = os.path.splitext(item_folder)
            if folder_name.isdigit():
                existing_ids.append(int(folder_name))

        if not existing_ids:
            return 0

        return max(existing_ids) + 1

    def is_game_over(self):
        end_sign =self._get_item_position('buttons/continue.png')
        if end_sign.x == -1 and end_sign.y == -1:
            return False
        else:
            return True

    def reset(self):
        continue_sign = self._get_item_position('buttons/continue.png')
        if continue_sign.x != -1 and continue_sign.y != -1:
            pyautogui.moveTo(continue_sign.x, continue_sign.y)
            pyautogui.click()
            time.sleep(3)
        
        sail_sign = self._get_item_position('buttons/sail.png')
        if sail_sign.x != -1 and sail_sign.y != -1:
            pyautogui.moveTo(sail_sign.x, sail_sign.y)
            pyautogui.click()
            time.sleep(3)

test = False
if test:
    game = TripleTownHandler()
    gameSim = TripleTownSim()
    while True:
        game_status = game.game_status()
        gameSim.display_board(game_status)
        action = int(input("请输入动作:"))
        game.save_image(game.game_shot, action)
        if action == -1:
            break
        else:
            game.click_slot(action)