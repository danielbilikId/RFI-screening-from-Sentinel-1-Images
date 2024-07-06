import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 

import sys
sys.path.insert(1, '/utils/')
from  utils.load_image import load_image

def get_background_image(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError("No image files found in the specified directory.")

    first_image = np.array((load_image(os.path.join(directory, image_files[0]))))
    height, width = first_image.shape

    image_stack = np.zeros((height, width), dtype=np.float64)
    
    for image_file in image_files:
        img_path = os.path.join(directory, image_file)
        img_array = load_image(img_path)
        if img_array.shape != (height, width):
            print(f"Skipping {image_file} due to mismatched dimensions.")
            continue
        
        image_stack += img_array
        print(f"Processed: {image_file}")

    avg_image = image_stack / len(image_files)
    
    return avg_image

def save_and_display_result(avg_image, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    #cv2.imwrite(os.path.join(output_directory, "Beijing_background_image.png"), avg_image)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(avg_image, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title('RFI-free Background Image')
    plt.axis('off')
    plt.show()

input_directory = "./Beijing_pngs_registered"
output_directory = "./Background_images"

avg_image = get_background_image(input_directory)
save_and_display_result(avg_image, output_directory)
