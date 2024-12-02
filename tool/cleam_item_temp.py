import os
import cv2
import heapq
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim

folder_path = "item_template/0" # image folder path
protected_number = 40 # protected file number
last_n = 50 # last n images

def calculate_similarity(img1_path, img2_path):
    # get the images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0  # if the image is not found, return 0

    # arrange the images shape
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # calculate the similarity ssim
    score, _ = compare_ssim(img1, img2, full=True)
    return score

def find_least_similar_images(folder_path, last_n=50, protected_files=None):
    # get all the images in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    
    if len(image_files) < 2:
        raise ValueError("Not enough images to compare.")

    results = {}

    for target_image_path in tqdm(image_files, desc="Processing images"):
        similarity_scores = []

        for compare_image_path in image_files:
            if target_image_path == compare_image_path:
                continue
            score = calculate_similarity(target_image_path, compare_image_path)
            print(f"score: {score}")
            similarity_scores.append(score)

        # calculate the average similarity score
        if similarity_scores:
            average_score = sum(similarity_scores) / len(similarity_scores)
        else:
            average_score = 0

        results[target_image_path] = average_score

    # exclude protected files
    if protected_files:
        results = {path: score for path, score in results.items() if os.path.basename(path) not in protected_files}

    # get the least similar images
    least_similar_images = sorted(results.items(), key=lambda x: x[1])[:last_n]
    return [item[0] for item in least_similar_images]

def main():
    protected_files = [f"{i}.png" for i in range(protected_number)]  # get the protected files

    try:
        least_similar_images = find_least_similar_images(folder_path, last_n, protected_files)

        for image_path in os.listdir(folder_path):
            full_path = os.path.join(folder_path, image_path)

            if image_path in protected_files:
                print(f"skip protected_files: {image_path}")
                continue

            if full_path not in least_similar_images:
                try:
                    # os.remove(full_path)
                    print(f"remove: {full_path}")
                except Exception as e:
                    print(f"can't remove {full_path}: {e}")

    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()
