import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_filename = '2apbkfh.jpg'
image_path = os.path.join('.', 'mark_pic', image_filename)

# read image
img = cv2.imread(image_path)
if img is None:
    print(f"Image not found, please check the path: {image_path}")
    exit()

# convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert BGR to HSV for processing
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. set initial HSV thresholds
# this is a rough range for dark defect areas, you can tune it later
lower_bound = np.array([0, 40, 80])
upper_bound = np.array([25, 255, 180])

# 3. get the initial binary mask
raw_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

# 4. morphology for denoising
kernel = np.ones((5, 5), np.uint8)
# opening: remove small white noise
clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
# closing: fill small holes inside defect regions
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

# 5. visualize results with matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title(f'1. Original: {image_filename}')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('2. Raw HSV Mask')
plt.imshow(raw_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('3. Clean Mask (Morphology)')
plt.imshow(clean_mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()