import os
import cv2
import heapq
from tqdm import tqdm

def calculate_similarity(img1_path, img2_path):
    target_image = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    compare_image = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if target_image is None or compare_image is None:
        return 0
    result = cv2.matchTemplate(target_image, compare_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val

def find_top_similar_images(folder_path, top_n=50):
    """
    找出每张图片与其他图片最相似的前 top_n 张图片。
    """
    # 获取文件夹中所有图片路径
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    if len(image_files) < 2:
        raise ValueError("文件夹中图片数量不足以计算相似度。")

    results = {}

    for target_image_path in tqdm(image_files, desc="Processing images"):

        # 保存相似度结果
        similarity_scores = []

        for compare_image_path in image_files:
            if target_image_path == compare_image_path:
                continue

            score = calculate_similarity(target_image_path, compare_image_path)
            similarity_scores.append(score)

        if len(similarity_scores) > 0:
            average_score = sum(similarity_scores) / len(similarity_scores)
        else:
            average_score = 0
        results[target_image_path] = average_score

    return results

def main():
    folder_path = "item_template/0"   # 替换为图片文件夹路径
    top_n = 50  # 需要保留的最相似图片数量

    try:
        results = find_top_similar_images(folder_path, top_n)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_50_results = sorted_results[:50]
        top_50_paths = {item[0] for item in top_50_results}  # 使用集合方便比对
        final_50_results = sorted_results[-50:]
        final_50_paths = {item[0] for item in final_50_results}  # 使用集合方便比对

        protected_files = [f"{i}.png" for i in range(11)]  # 生成保护文件列表
        for image_path in results.keys():
            # 检查文件是否在保护文件列表中
            if os.path.basename(image_path) in protected_files:
                print(f"跳过保护文件: {image_path}")
                continue

            if image_path not in top_50_paths and image_path not in final_50_paths:
                try:
                    os.remove(image_path)
                    print(f"已删除: {image_path}")
                except Exception as e:
                    print(f"无法删除 {image_path}: {e}")
    
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
