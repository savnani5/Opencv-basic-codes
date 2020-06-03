import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype = "uint8")
green = (255, 255, 0)
blue = (255, 0, 0)
cv2.line(canvas, (0, 0), (300, 300), green)

cv2.rectangle(canvas, (10, 10), (60, 60), blue,-1)
cv2.imshow("Canvas", canvas)

cv2.waitKey(0)
red = (0, 0, 255)
cv2.rectangle(canvas, (200, 50), (160, 160), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''image = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\21.jpeg',1)

corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)
image[0:100, 0:100] = (255, 0, 0)
cv2.imshow("Updated", image)
cv2.waitKey(0)'''


'''cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

image[0, 0] = (255, 0, 0)
(b, g, r) = image[0, 0]

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))'''

