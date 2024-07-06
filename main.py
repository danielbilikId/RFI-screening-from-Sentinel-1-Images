#40.011086,116.257454 40.406605,119.203476 38.414730,119.614365 38.017380,116.750717 40.011086,116.257454
import os 
import numpy as np
from skimage import io, filters, morphology
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 
from utils.load_image import load_image
def load_images(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):  # Assuming images are PNG format
            img = load_image(os.path.join(directory, filename))
            images.append(img)
    return images

# Function to preprocess images and generate difference images
def preprocess_and_detect(images, background_image):
    background_image = background_image.astype(np.uint8) 
    results = []
    for img in tqdm(images):
        # Compute log-ratio of current image (Iu) and background image (Ib)
        img = np.where(img == 0, 1e-8, img).astype(np.uint8)  # Replace zeros with a small value
        min_val = np.min([img.min(), background_image.min()])  # Find minimum of both images
        max_val = np.max([img.max(), background_image.max()])  # Find maximum of both images
        img_normalized = (img - min_val) / (max_val - min_val)  # Normalize img
        background_image_normalized = (background_image - min_val) / (max_val - min_val)  # Normalize background_image
        diff_image_1 = np.log((img_normalized + 1) / (background_image_normalized + 1))
        np.where(diff_image_1 < 0, 0, diff_image_1)
        min_val = diff_image_1.min()  # Find minimum of both images
        max_val = diff_image_1.max()# Find maximum of both images
        diff_image = (diff_image_1 - min_val) / (max_val - min_val)
        diff_image = np.clip(diff_image_1, a_min=0, a_max=150)
        min_val = diff_image.min()  # Find minimum of both images
        max_val = diff_image.max()# Find maximum of both images
        diff_image = (diff_image - min_val) / (max_val - min_val)
        threshold_value = filters.threshold_otsu(diff_image)
        thresh_img = diff_image > threshold_value  # Convert to binary image (foreground: True, background: False)

        # Thresholding using Otsu's method


        # Morphological filtering to remove speckle noise
        filtered_image = morphology.remove_small_objects(thresh_img, min_size=50)
        
        # Calculate entropy of the difference image
        #entropy_value = filters.rank.entropy(diff_image, morphology.disk(5))
        
        # K-means clustering to classify images
        reshaped_image = diff_image.reshape((-1, 1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(reshaped_image)
        labels = kmeans.labels_.reshape(diff_image.shape)
        
        # Store results
        results.append({
            'diff_image': diff_image,
            'segmented_image': thresh_img,
            'filtered_image': filtered_image,
            #'entropy_value': entropy_value,
            'cluster_labels': labels
        })
    
    return results

# Function to display images
def display_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Directory containing SAR images
    images_directory = './Beijing_pngs_registered'
    
    # Load images
    images = load_images(images_directory)
    
    # Assuming background image Ib is already obtained
    background_image = io.imread('./Background_images/Beijing_background_image.png', as_gray=True)
    
    # Preprocess images and detect RFI
    detection_results = preprocess_and_detect(images, background_image)
    
    # Display images after each filtering step
    for idx, result in enumerate(detection_results):
        diff_image = result['diff_image']
        segmented_image = result['segmented_image']
        filtered_image = result['filtered_image']
        
        # Display images
        display_images([diff_image, segmented_image, filtered_image],
                       ['Log-ratio Image', 'Segmented Image (Otsu)', 'RF signature Image'])

# # Path to the TIFF image
# path = '/Users/igalbilik/Downloads/S1A_IW_GRDH_1SDV_20210703T102305_20210703T102330_038612_048E5B_7B68.SAFE/measurement/s1a-iw-grd-vh-20210703t102305-20210703t102330-038612-048e5b-002.tiff'

# # Open the TIFF file
# src = rasterio.open(path)

# # Read the image data
# image = src.read(1)  # Assuming it's a single-band image

# # Plot the image using matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
# plt.colorbar(label='Intensity')
# plt.title('Sentinel-1 SAR Image')
# plt.axis('off')
# plt.show()

# # Close the rasterio dataset
# src.close()

