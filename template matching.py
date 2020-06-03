import cv2
import numpy as np

template_data=[]
img = np.zeros((100,100,3), np.uint8)

cv2.rectangle(img,(25,30),(60,70),(255,255,255),-1)

rows,cols,ch = img.shape

for i in range(0,360,20):
    M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    template_data.append(dst)
    
    #cv2.imshow('Result',dst)
    #cv2.waitKey(0)

test_image = cv2.imread('C:\\Users\\HP\\Desktop\\Fero\\21.jpeg',1)

hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([110, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(test_image,test_image, mask= mask)

cv2.imshow('res',res)  
  
for tmp in template_data:
        (tH, tW) = tmp.shape[:2]
        cv2.imshow("Template", tmp)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        result = cv2.matchTemplate(test_image, tmp, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + tW, top_left[1] + tH)
        cv2.rectangle(test_image,top_left, bottom_right,255, 2)
        
cv2.imshow('Result',test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    