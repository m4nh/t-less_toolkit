import cv2
import sys
import numpy as np

image = cv2.imread(sys.argv[1])

gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
blurredgx = cv2.GaussianBlur(gx, (11, 3), 1)
blurredgy = cv2.GaussianBlur(gy, (11, 3), 1)
magnitude, angle = cv2.cartToPolar(blurredgx, blurredgy)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

gy = gy - np.min(gy)
gy = (gy / np.max(gy))*2.0 - 1.0
angle = angle / np.max(angle)
print(np.min(angle), np.max(angle))
cv2.imshow("img", angle)
cv2.waitKey(0)
