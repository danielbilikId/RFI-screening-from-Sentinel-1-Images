import os
import cv2
import rasterio
import numpy as np 

def resize_tiff_images(input_dir, output_dir, new_width=1024, new_height=1024, output_format='png'):
    """
    Resize TIFF images in the input directory and save them to the output directory in the specified format.
    Visualizes the resized images using matplotlib to ensure proper scaling.

    Args:
    - input_dir (str): Directory path containing TIFF files.
    - output_dir (str): Directory path to save resized images.
    - new_width (int, optional): Width of the resized image (default: 1024).
    - new_height (int, optional): Height of the resized image (default: 1024).
    - output_format (str, optional): Output image format ('jpg', 'png', etc.). Default is 'png'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            tiff_path = os.path.join(input_dir, filename)

            with rasterio.open(tiff_path) as src:
                image = src.read(1)
            
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            vmin = np.percentile(resized_image, 1)
            vmax = np.percentile(resized_image, 99)
            
            clipped_image = np.clip(image, vmin, vmax)
    
            normalized_image = (clipped_image - vmin) / (vmax - vmin) * 255
            normalized_image = normalized_image.astype(np.uint8)
    
            
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.' + output_format)
 
            cv2.imwrite(output_path, normalized_image)

input_dir = './TIFFS_LATE'
output_dir = './pngs_late'

# Resize TIFF images
resize_tiff_images(input_dir, output_dir, new_width=7024, new_height=7024,output_format='png')