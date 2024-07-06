import os
import cv2
import numpy as np

def preprocess_image(image):
    return cv2.GaussianBlur(image, (1, 1), 0)

def image_registration(fixed_image, moving_image):
    fixed_image = preprocess_image(fixed_image)
    moving_image = preprocess_image(moving_image)

    sift = cv2.SIFT_create(nfeatures=5000)
    keypoints1, descriptors1 = sift.detectAndCompute(fixed_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(moving_image, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    height, width = fixed_image.shape
    aligned_moving_image = cv2.warpPerspective(moving_image, h, (width, height))

    return aligned_moving_image

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.resize(img,(2000,1064))

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the input folder.")
        return
    
    reference_file = os.path.join(input_folder, image_files[0])
    reference_img = load_image(reference_file)
    
    cv2.imwrite(os.path.join(output_folder, f"reference_{image_files[0]}"), reference_img)
    print(f"Saved reference image: {image_files[0]}")
    
    for image_file in image_files[1:]:
        input_file = os.path.join(input_folder, image_file)
        
        img = load_image(input_file)
        
        registered_img = image_registration(reference_img, img)
        
        output_file = os.path.join(output_folder, f"registered_{image_file}")
        cv2.imwrite(output_file, registered_img)
        
        print(f"Processed and saved: {image_file}")

input_folder = "./Beijing_pngs"
output_folder = "./Beijing_pngs_registered"
process_images(input_folder, output_folder)