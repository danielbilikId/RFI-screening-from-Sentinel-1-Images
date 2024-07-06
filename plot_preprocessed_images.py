import os
import cv2
import math
import matplotlib.pyplot as plt

def load_and_resize(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(2000,1024))
    return img

def visualize_comparison(input_folder, output_folder):
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    n_images = len(input_files)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    
    for i, input_file in enumerate(input_files, 1):
        input_path = os.path.join(input_folder, input_file)
        output_path = os.path.join(output_folder, f"registered_{input_file}")
        
        if not os.path.exists(output_path):
            print(f"Processed file not found for {input_file}")
            continue
        
        original_img = load_and_resize(input_path)
        processed_img = load_and_resize(output_path)
        
        ax1 = fig.add_subplot(n_rows, n_cols * 2, i * 2 - 1)
        ax1.imshow(original_img)
        ax1.set_title(f'Original', fontsize=8)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(n_rows, n_cols * 2, i * 2)
        ax2.imshow(processed_img)
        ax2.set_title(f'Processed', fontsize=8)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Comparison of Original and Processed Images', fontsize=16)
    plt.subplots_adjust(top=0.95) 
    plt.show()

input_folder = "./Beijing_pngs"
output_folder = "./Beijing_pngs_registered"
visualize_comparison(input_folder, output_folder)