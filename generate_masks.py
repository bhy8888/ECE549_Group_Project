import json
import cv2
import numpy as np
import os
import glob

data_dir = './mark_pic' 

# find all image files in the folder (jpg, jpeg, png)
image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_files.extend(glob.glob(os.path.join(data_dir, ext)))

for img_path in image_files:
    # read image to get height and width
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_shape = img.shape[:2]  # (height, width)
    
    # create a black mask (0 = normal area)
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    # find corresponding json file
    base_name = os.path.splitext(img_path)[0]
    json_path = base_name + '.json'
    
    # if json exists, mark defect area as white (255)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data['shapes']:
            if shape['label'] == 'defect':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
        print(f"Processing defect image: Mask generated -> {os.path.basename(img_path)}")
    else:
        # no json → keep mask black
        print(f"Processing normal image (no JSON): Black mask -> {os.path.basename(img_path)}")
    
    # save mask image with suffix _mask.png
    mask_filename = base_name + '_mask.png'
    cv2.imwrite(mask_filename, mask)

print("\nAll mask generation completed!")