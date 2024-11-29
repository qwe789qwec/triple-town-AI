import pyautogui
import time
import os
import cv2
import numpy as np
from PIL import Image
import easyocr
from pathlib import Path

reader = easyocr.Reader(['en'], gpu = True)

screen_w, screen_h = 802, 639
score_x, score_y, score_w, score_h = 100, 40, 380, 50
game_area_x, game_area_y, game_area_w, game_area_h = 35, 90, 512, 512

class playgame:
    def __init__(self, save_dir='gameplay'):
        self.step = 0
        self.save_dir = save_dir
        self.game_number = self.get_next_game_number()
        self.screen_x, self.screen_y = self.get_window_info()
        self.mouse_x = self.screen_x + 533
        self.mouse_y = self.screen_y + 108

        self.mouse_game_x = self.screen_x + 70
        self.mouse_game_y = self.screen_y + 160
        self.end_x = self.screen_x + 682
        self.end_y = self.screen_y + 247
        self.start_x = self.screen_x + 97
        self.start_y = self.screen_y + 198

        self.slot_gap = 80
        self.slot_size = 60

        self.last_number = 0

    def get_window_info(self):
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        template = cv2.imread('window_template.png', cv2.IMREAD_COLOR)
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # get window position
        if max_val > 0.8:  # matching threshold
            print(f"window position: {max_loc}")
            return max_loc[0], max_loc[1]
        else:
            print("no matching window found!")
            return -1, -1
        

    def get_latest_image(self, folder_path = '/home/wilson/Pictures/Screenshots'):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not image_files:
            print("could not find any image files in the folder")
            return None
        
        latest_image = max(image_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        latest_image_path = os.path.join(folder_path, latest_image)
        image = cv2.imread(latest_image_path)
        # os.remove(latest_image_path)
        return image

    def get_next_game_number(self):
        os.makedirs(self.save_dir, exist_ok=True)
        existing_files = os.listdir(self.save_dir)

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
    
    def take_screenshot(self):
        pyautogui.moveTo(self.mouse_x, self.mouse_y)
        pyautogui.click()
        screenshot = pyautogui.screenshot(region=(self.screen_x, self.screen_y, screen_w, screen_h))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.latest_image = frame

    def save_image(self, image, action=-1):
        if image is None:
            print("No image to save.")
            return
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.step = self.step + 1
        now_score = self.get_score()
        filename = f"game_{self.game_number}_{self.step}_{action}.png"
        pil_image.save(os.path.join(self.save_dir, filename))

    def show_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('game', image_rgb)
        cv2.waitKey(0)
        time.sleep(1)
        cv2.destroyAllWindows()

    def get_score(self):
        score_region = self.latest_image[score_y:score_y + score_h, score_x:score_x + score_w]
        gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # score = pytesseract.image_to_string(thresh_image, config='outputbase digits')
        # self.save_image(thresh_image)
        score = reader.readtext(thresh_image)
        # self.show_image(thresh_image)
        # if not score:
        #     self.save_image(thresh_image)
        #     self.show_image(thresh_image)
        #     return 0
        if not score:
            return None  # return the last known score
        score_str = score[0][1]  # get the first detected text
        try:
            if score_str.count(',') > 0:
                score_number = int(score[0][1].replace(',', ''))
            else:
                score_number = int(score[0][1])
            self.last_number = score_number
        except ValueError:
            return self.last_number
        return score_number
    
    def slot_region(self, pos_x, pos_y):
        slot_x = 70 + self.slot_gap * pos_x - (self.slot_size / 2)
        slot_y = 160 + self.slot_gap * pos_y - (self.slot_size / 2)
        return int(slot_x), int(slot_y)


    def get_game_area(self):
        check_time = 0
        while True:
            slot_matrix = np.full((6, 6), -1)
            invalid_detected = False
            for slot in range(36):
                row, col = divmod(slot, 6)
                slot_x, slot_y = self.slot_region(row, col)
                slot_img = self.latest_image[slot_y:slot_y + self.slot_size, slot_x:slot_x + self.slot_size]
                index = self.find_matching_item(slot_img)
                # print(f"slot {slot} matched with item {index}")
                slot_matrix[col, row] = index
                if index == 21 and check_time <= 1:
                    time.sleep(1.5)
                    print(f"Invalid index {index} detected. Retrying in 1 seconds...")
                    invalid_detected = True
                    break
            if invalid_detected:
                check_time += 1
                self.take_screenshot()
                continue
            else:
                break

        next_item = self.latest_image[85:85 + 60, 508:508 + 60]
        next_item_id = self.find_matching_item(next_item)
        return slot_matrix, next_item_id
        # self.save_image(next_item, 37)
        # game_area = self.latest_image[game_area_y:game_area_y + game_area_h, game_area_x:game_area_x + game_area_w]
        # gray_game_area = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
        # return game_area

    def slot_with_item(self, slot, item):

        result = np.full((7, 7), 0)
        result[:3, :3] = slot[:3, :3]   # top_left
        result[:3, 4:] = slot[:3, 3:]   # top_right
        result[4:, :3] = slot[3:, :3]   # bottom_left
        result[4:, 4:] = slot[3:, 3:]   # bottom_right
        result[3, 3] = item

        return result
    
    def split_result(self, result):

        slot_matrix = np.full((6, 6), -1)
        slot_matrix[:3, :3] = result[:3, :3]
        slot_matrix[:3, 3:] = result[:3, 4:]
        slot_matrix[3:, :3] = result[4:, :3]
        slot_matrix[3:, 3:] = result[4:, 4:]
        item = result[3, 3]

        return slot_matrix, item

    def find_matching_item(self, item_image):
        max_match_value = 0.6
        matching_item_id = None
        
        for item_folder in os.listdir("item_template"):
            item_folder_path = os.path.join("item_template", item_folder)
            
            if os.path.isdir(item_folder_path):
                for template_file in os.listdir(item_folder_path):
                    template_image_path = os.path.join(item_folder_path, template_file)
                    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
                    if template_image is None:
                        continue
                    
                    item_image_gray = cv2.cvtColor(item_image, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(item_image_gray, template_image, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > max_match_value:
                        max_match_value = max_val
                        matching_item_id = item_folder
                    if max_val > 0.8 and matching_item_id is not None:
                        if not template_file.startswith("0_"):
                            new_name = f"0_{template_file}"
                            new_path = os.path.join(item_folder_path, new_name)
                            os.rename(template_image_path, new_path)  # 修改文件名
                            print(f"Renamed: {template_image_path} -> {new_path}")
                        break

        if matching_item_id is None:
            print("No match found, creating a new folder for this item.")
            time.sleep(3)
            new_item_id = str(self.get_next_path_id("item_template"))
            new_folder = os.path.join("item_template", new_item_id)
            new_folder = Path(new_folder)
            new_folder.mkdir(parents=True, exist_ok=True)
            image_index = self.get_next_path_id(new_folder)
            cv2.imwrite(os.path.join("item_template", new_item_id, f"{image_index}.png"), item_image)
            matching_item_id = new_item_id
        else:
            image_index = self.get_next_path_id(os.path.join("item_template", matching_item_id))
            if image_index < 500 and max_match_value < 0.8:
                cv2.imwrite(os.path.join("item_template", matching_item_id, f"{image_index}.png"), item_image)

        return int(matching_item_id)
    
    def get_next_path_id(self, folder_path):
        existing_ids = []
        for item_folder in os.listdir(folder_path):
            folder_name, _ = os.path.splitext(item_folder)
            if folder_name.isdigit():
                existing_ids.append(int(folder_name))

        if not existing_ids:
            return 0

        return max(existing_ids) + 1

    
    def is_game_end(self):
        template = cv2.imread('end_game_template.png', cv2.IMREAD_COLOR)
        result = cv2.matchTemplate(self.latest_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # get window position
        if max_val > 0.8:  # matching threshold
            print(f"end game position: {max_loc}")
            print(max_val)
            return True
        else:
            # print("no matching window found!")
            return False
        
    def mouse_click(self, pos_onehot):
        # pos = pos + 1
        # x = pos % 6
        # y = pos // 6

        pos = pos_onehot.max(0).indices
        pos_number = pos.item()
        self.save_image(self.latest_image, pos_number)

        row, col = divmod(pos_number, 6)
        pyautogui.moveTo(self.mouse_game_x + self.slot_gap * col, self.mouse_game_y + self.slot_gap * row)
        pyautogui.click()
        if pos_number in {2, 3}:
            time.sleep(5.0)
        else:
            time.sleep(1.0)


    def restart_game(self):
        pyautogui.click(self.end_x, self.end_y)
        time.sleep(3)
        pyautogui.click(self.start_x, self.start_y)
        time.sleep(5)

# gamesc = GameScreen()

# for i in range(3):
#     gamesc.take_screenshot()
#     # gamesc.show_image(gamesc.latest_image)
#     gamesc.save_image(gamesc.latest_image, 12)
#     state , next_item = gamesc.get_game_area()
#     print(state)
#     print(next_item)
#     print(gamesc.get_score())
#     time.sleep(1)