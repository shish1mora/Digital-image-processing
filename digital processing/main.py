import cv2
import numpy as np
import os

def save_image(image, filename):
    cv2.imwrite(filename, image)


def binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary


def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def skeletonize_image(binary_image):

    skeleton = np.zeros(binary_image.shape, dtype=np.uint8)
    while True:
        eroded = cv2.erode(binary_image, np.ones((3, 3), np.uint8))
        temp = cv2.dilate(eroded, np.ones((3, 3), np.uint8))
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()
        if cv2.countNonZero(binary_image) == 0:
            break
    return skeleton


def segmentation_and_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_image = np.zeros_like(image)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 1)
    return segmented_image

def main():
    input_image_path = 'Z:/pgnpython/eye_image.jpg'
    image = cv2.imread(input_image_path)

    if image is None:
        print("Ошибка: не удалось загрузить изображение. Проверьте путь.")
        return

    output_folder = os.path.dirname(input_image_path)

    binary_image = binarization(image)
    save_image(binary_image, os.path.join(output_folder, 'binary_image.jpg'))

    enhanced_image = enhance_contrast(image)
    save_image(enhanced_image, os.path.join(output_folder, 'enhanced_image.jpg'))

    skeleton_image = skeletonize_image(binary_image) 
    save_image(skeleton_image, os.path.join(output_folder, 'skeleton_image.jpg'))

    segmented_image = segmentation_and_contours(image)
    save_image(segmented_image, os.path.join(output_folder, 'segmented_image.jpg'))

if __name__ == "__main__":
    main()

