import os
import cv2
import numpy as np

DATASET_PATH = r"C:\Users\LENOVO\face_partial_project\data\database\lfw"

valid_count = 0
broken_count = 0
dark_count = 0

broken_files = []
dark_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)

            img = cv2.imread(img_path)

            # الصورة لم تُقرأ
            if img is None:
                broken_count += 1
                broken_files.append(img_path)
                print(f"[BROKEN] {img_path}")
                continue

            valid_count += 1

            # متوسط الإضاءة
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)

            # إذا كانت داكنة جدًا
            if mean_val < 10:
                dark_count += 1
                dark_files.append((img_path, mean_val))
                print(f"[DARK] {img_path} --> mean={mean_val:.2f}")

print("\n========== SUMMARY ==========")
print(f"Valid images : {valid_count}")
print(f"Broken images: {broken_count}")
print(f"Dark images  : {dark_count}")

print("\n========== BROKEN FILES ==========")
for path in broken_files[:20]:
    print(path)

print("\n========== DARK FILES ==========")
for path, val in dark_files[:20]:
    print(f"{path} --> mean={val:.2f}")