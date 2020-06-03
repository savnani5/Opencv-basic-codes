import cv2
import numpy as np
import imutils

image = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\21.jpeg',1)
img = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\image with laser spots.png',1)

capture = cv2.addWeighted(image,0.2, img,0.8,0)

cv2.imshow('capture', capture)
cv2.waitKey(0)
cv2.destroyAllWindows()

capture1 = img + image
cv2.imshow('capture1', capture1)
cv2.waitKey(0)
cv2.destroyAllWindows()