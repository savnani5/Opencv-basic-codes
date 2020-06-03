import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([110, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #kernel = np.ones((15, 15), np.float32)/225
    #blur = cv2.GaussianBlur(res, (15, 15), 0)
    #median = cv2.medianBlur(res, 15)
    #bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    #edges = cv2.Canny(frame, 100, 100)

    #cv2.imshow('frame',frame)
    #cv2.imshow(' median', median)
    #cv2.imshow('blur', blur)
    cv2.imshow('res',res)
    #cv2.imshow('bilateral', bilateral)
    #cv2.imshow('edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()