import cv2
import numpy as np
import os
import glob

def calculate_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks"""
    # make sure masks are binary (0 or 1)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        return intersection / union

# 1. set test directory
test_dir = './mark_pic'

# get all ground truth mask files
gt_mask_files = glob.glob(os.path.join(test_dir, '*_mask.png'))

total_iou = 0.0
valid_images_count = 0

print("Start evaluating traditional CV baseline...\n")

for gt_path in gt_mask_files:
    # infer original image path
    base_name = gt_path.replace('_mask.png', '')
    
    # try possible image extensions
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        if os.path.exists(base_name + ext):
            img_path = base_name + ext
            break
            
    if img_path is None:
        continue
        
    # read image and ground truth
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # ---------------------------------------------------------
    # your OpenCV baseline method
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 20])
    upper_bound = np.array([25, 255, 180])
    raw_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    # ---------------------------------------------------------
    
    # compute IoU
    iou_score = calculate_iou(clean_mask, gt_mask)
    total_iou += iou_score
    valid_images_count += 1
    
    # print first few results
    if valid_images_count <= 5:
        print(f"File: {os.path.basename(img_path)} - IoU: {iou_score:.4f}")

# compute mean IoU
if valid_images_count > 0:
    mean_iou = total_iou / valid_images_count
    print(f"\nEvaluation done! Tested {valid_images_count} images.")
    print(f"Mean IoU (traditional CV baseline): {mean_iou:.4f}")
else:
    print("No matching images and masks found.")