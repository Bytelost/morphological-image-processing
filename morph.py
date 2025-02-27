import cv2
import numpy as np
import os

# 
def binarize_image(image_path, threshold=128):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def pad_image(image, kernel):
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def erosion(image, kernel):
    padded = pad_image(image, kernel)
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel == kernel * 255):
                result[i, j] = 255
    
    return result

def dilation(image, kernel):
    padded = pad_image(image, kernel)
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.any(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel == 255):
                result[i, j] = 255
    
    return result

def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)

# Exemplo de uso
if __name__ == "__main__":
    image_path = "image.png"
    binary_image = binarize_image(image_path)
    
    # Create a folder for the results
    results = "results"
    os.makedirs(results, exist_ok=True)
    
    # Create the kernel to determine the neighborhood of the image
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    eroded = erosion(binary_image, kernel)
    dilated = dilation(binary_image, kernel)
    opened = opening(binary_image, kernel)
    closed = closing(binary_image, kernel)
    
    # Save results in the folder
    cv2.imwrite(os.path.join(results, "eroded.png"), eroded)
    cv2.imwrite(os.path.join(results, "dilated.png"), dilated)
    cv2.imwrite(os.path.join(results, "opened.png"), opened)
    cv2.imwrite(os.path.join(results, "closed.png"), closed)
    