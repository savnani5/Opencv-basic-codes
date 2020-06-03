import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\query_image.png',0)          # queryImage
img2 = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\Capture.png',0)              # trainImage

orb = cv2.ORB_create(nfeatures=100000)


kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)


if len(matches)>1050:
    
# Match descriptors.

    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)

    cv2.imshow('img3', img3)

while(1):
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break    
    
#plt.show()
