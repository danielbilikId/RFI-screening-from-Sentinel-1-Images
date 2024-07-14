import os
import numpy as np
from skimage import io, filters, morphology, measure
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.load_image import load_image

def load_images(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):  
            img = load_image(os.path.join(directory, filename))
            images.append(img)
    return images

def preprocess_images(images, background_image):
    background_image = background_image.astype(np.uint8)
    diff_images = []
    
    for img in tqdm(images):
        img = np.where(img == 0, 1e-8, img).astype(np.uint8)
        min_val = np.min([img.min(), background_image.min()])
        max_val = np.max([img.max(), background_image.max()])
        img_normalized = (img - min_val) / (max_val - min_val)
        background_image_normalized = (background_image - min_val) / (max_val - min_val)
        diff_image_1 = np.log((img_normalized + 1) / (background_image_normalized + 1))
        diff_image_1 = np.where(diff_image_1 < 0, 0, diff_image_1)
        min_val = diff_image_1.min()
        max_val = diff_image_1.max()
        diff_image = (diff_image_1 - min_val) / (max_val - min_val)
        diff_image = np.clip(diff_image_1, a_min=0, a_max=150)
        diff_image = (diff_image - min_val) / (max_val - min_val)
        print(np.shape(diff_image))
        diff_images.append(diff_image)
    
    return diff_images

def classify_images_kmeans(diff_images):
    combined_diff_images = np.array([img.flatten() for img in diff_images])
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(combined_diff_images)
    
    return kmeans.labels_

def detect_rfi(diff_images):
    results = []
    
    for diff_image in diff_images:
        threshold_value = filters.threshold_otsu(diff_image)
        thresh_img = diff_image > threshold_value
        filtered_image = morphology.remove_small_objects(thresh_img, min_size=50)
        entropy_value = measure.shannon_entropy(filtered_image)

        results.append({
            'diff_image': diff_image,
            'segmented_image': thresh_img,
            'filtered_image': filtered_image,
            'entropy_value': entropy_value
        })
    
    return results

def display_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images_directory = './Beijing_pngs_registered'
    images = load_images(images_directory)

    background_image = io.imread('./Background_images/Beijing_background_image.png', as_gray=True)
    diff_images = preprocess_images(images, background_image)
    detection_results = detect_rfi(diff_images)
    
    entropies = [result['entropy_value'] for result in detection_results]
    entropy_threshold = np.mean(entropies)
    classifications = classify_images_kmeans(diff_images)
    
    for idx, (result, classification) in enumerate(zip(detection_results, classifications)):
        diff_image = result['diff_image']
        segmented_image = result['segmented_image']
        filtered_image = result['filtered_image']
        entropy = result['entropy_value']
        class_label = 'without RFI' if classification == 1 else 'with RFI'
        print("Entropy for image {}: {:.2f} - Predicted class: {}".format(idx, entropy, class_label))
        
        display_images([diff_image, segmented_image, filtered_image],
                       ['Log-ratio Image', 'Segmented Image (Otsu)', 'RF Signature Image'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(entropies, marker='o', linestyle='-', color='b')
    plt.axhline(y=entropy_threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Entropy Values of Images')
    plt.xlabel('Image Index')
    plt.ylabel('Entropy')
    plt.legend()
    plt.show()
