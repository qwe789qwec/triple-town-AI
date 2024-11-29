import pyautogui
import cv2
import numpy as np

def get_window_info():
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # window template
    template = cv2.imread('window_template.png', cv2.IMREAD_COLOR)

    # match template
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # get window position
    if max_val > 0.8:  # matching threshold
        print(f"window position: {max_loc}")
    else:
        print("no matching window found!")

    return max_loc[0], max_loc[1]

window_x, window_y = get_window_info()
print("window_x", window_x)
print("window_y", window_y)