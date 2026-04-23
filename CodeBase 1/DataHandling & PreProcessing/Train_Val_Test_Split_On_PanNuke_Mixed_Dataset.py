import os
import random
import shutil

# Paths
IMG_DIR = r"E:\404_Found\H&E Stained Histological Nucleus Instancve Segmentation by YOLOv8x\Dataset\PanNuke_Chaotic_Mixed_Dataset\images"
MASK_DIR = r"E:\404_Found\H&E Stained Histological Nucleus Instancve Segmentation by YOLOv8x\Dataset\PanNuke_Chaotic_Mixed_Dataset\masks"

OUTPUT_DIR = r"E:\404_Found\H&E Stained Histological Nucleus Instancve Segmentation by YOLOv8x\Dataset\PanNuke_Dataset_Splitted"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Get all images
images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]

# Shuffle
random.seed(42)
random.shuffle(images)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15

n = len(images)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Function to copy
def copy_files(img_list, split):
    for img_name in img_list:
        img_path = os.path.join(IMG_DIR, img_name)
        mask_name = img_name.replace(".png", ".png")  # adjust if needed
        mask_path = os.path.join(MASK_DIR, mask_name)

        shutil.copy(img_path, os.path.join(OUTPUT_DIR, split, "images", img_name))
        shutil.copy(mask_path, os.path.join(OUTPUT_DIR, split, "labels", mask_name))

# Copy files
copy_files(train_imgs, "train")
copy_files(val_imgs, "val")
copy_files(test_imgs, "test")

print("Dataset split complete.")