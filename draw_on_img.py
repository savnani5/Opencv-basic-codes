import cv2
import numpy as np

image = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\21.jpeg',1)

cv2.line(image, (0, 250), (200, 300), (255, 0, 0), 2)
cv2.rectangle(image, (0, 250), (200, 300), (0, 255, 0), 2 )
cv2.circle(image, (500,500), 200, (125, 255, 0), 3)

pts = np.array([[500, 20], [340, 200], [10, 600], [20, 40]])
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, (0, 0, 255), 2)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(image, 'Hello', (200, 500), font, 1, (255,0,0), 2, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

