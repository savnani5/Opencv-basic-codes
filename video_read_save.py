import cv2
import numpy as np

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter('C:\\Users\\HP\\Desktop\\Fero\\output1.avi', fourcc, 20.0, (640, 480))
out2 = cv2.VideoWriter('C:\\Users\\HP\\Desktop\\Fero\\output2.avi', fourcc, 20.0, (640, 480))

while(1):
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    out1.write(frame1)
    out2.write(frame2)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()


