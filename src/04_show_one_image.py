import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = r"C:\Users\LENOVO\face_partial_project\data\database\lfw\Bo_Pelini\Bo_Pelini_1.jpg"

img = cv2.imread(img_path)

print("Shape:", img.shape)
print("Min pixel:", img.min())
print("Max pixel:", img.max())
print("Mean pixel:", img.mean())

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Bo_Pelini_1")
plt.axis("off")
plt.show()