import cv2
import os
import numpy as np

# 定義兩個資料夾路徑
image_folder = 'gameplay'  # 需要比對的圖片所在資料夾
template_folder = 'item_template/21'  # 用來比對的模板圖片所在資料夾

# 取得資料夾中的所有圖片檔案
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
template_files = [f for f in os.listdir(template_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 讀取模板圖片
for template_file in template_files:
    template_path = os.path.join(template_folder, template_file)
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # 以灰階模式讀取模板
    if template_image is None:
        continue  # 如果模板讀取失敗，跳過

    # 遍歷圖像資料夾中的每一張圖片
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰階模式讀取圖片

        if image is None:
            continue  # 如果圖片讀取失敗，跳過

        # 使用模板比對
        result = cv2.matchTemplate(image, template_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 如果匹配值大於某個閾值，則認為匹配成功
        if max_val > 0.8:  # 0.8 只是示範閾值，你可以根據需要調整
            try:
                os.remove(image_path)
                print(f"已删除: {image_path}")
            except Exception as e:
                print(f"无法删除 {image_path}: {e}")
