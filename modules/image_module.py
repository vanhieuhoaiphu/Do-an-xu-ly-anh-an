import cv2, numpy as np

def rotateImage(main):
    rows, cols, steps = main.new_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), main.rotateValue, 1)  # thay đổi chiều của ảnh
    main.new_image = cv2.warpAffine(main.new_image, M, (cols, rows))

def gamma(img, gamma, c):
    return float(c) * pow(img, float(gamma));

def shearing(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M, (cols, rows))

    return img