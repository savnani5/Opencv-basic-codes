import cv2
import numpy as np
import objloader
import os
import math
# import pandas as pd

# def projection_matrix(camera_parameters, homography):

#     homography = homography * (-1)
#     rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
#     col_1 = rot_and_transl[:, 0]
#     col_2 = rot_and_transl[:, 1]
#     col_3 = rot_and_transl[:, 2]
#     # normalise vectors
#     l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
#     rot_1 = col_1 / l
#     rot_2 = col_2 / l
#     translation = col_3 / l
#     # compute the orthonormal basis
#     c = rot_1 + rot_2
#     p = np.cross(rot_1, rot_2)
#     d = np.cross(c, p)
#     rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
#     rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
#     rot_3 = np.cross(rot_1, rot_2)
#     # finally, compute the 3D projection matrix from the model to the current frame
#     projection = np.stack((rot_1, rot_2, rot_3, translation)).T
#     return np.dot(camera_parameters, projection)



# def render(img,vertices, projection, model, color=False):     
#     # vertices = vert
#     # print(type(vertices[10]))
#     # print(vertices[10])
#     # scale_matrix = np.eye(3) * 3
    

#    # for face in obj.faces:
#        # face_vertices = face[0]
#     # print('fghj')
#     # points = np.array([list((map(float,vertices[idx].split(' ')))) for idx,_ in enumerate(vertices)])
#     # points = np.array([vertices[idx].split(' ') for idx in range(6,len(vertices)+6)])
#     print(np.array([vertices[idx].split(' ') for idx in range(6,len(vertices)+6)]))
#     # render model in the middle of the reference surface. To do so,
#     # model points must be displaced
#     h, w = template_gray.shape
#     points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
#     dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
#     imgpts = np.int32(dst)
#    # if color is False:
#     cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
#     # else:
#     #     color = hex_to_rgb(face[-1])
#     #     color = color[::-1] # reverse
#     #     cv2.fillConvexPoly(img, imgpts, color)
#     cv2.imshow("Model",img)
#     return img



# def hex_to_rgb(hex_color):
#     """
#     Helper function to convert hex strings to RGB
#     """
#     hex_color = hex_color.lstrip('#')
#     h_len = len(hex_color)
#     return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))







# cont_ini = pd.read_csv("C:\\Users\\unnat\\OneDrive\\Desktop\\Other\\Fero\\Pose_Estimation\\40ft v1.obj",sep=" ")
# cont_ini = cont_ini.iloc[6:41100]
# cont_ini["XYZ"] = cont_ini["WaveFront"]+" "+cont_ini["*.obj"]+" "+cont_ini["file"] 


# content = open("C:\\Users\\unnat\\OneDrive\\Desktop\\Other\\Fero\\Pose_Estimation\\40ft v1.obj").read()
# # print(content)
# obj =  objloader.Obj.fromstring(content)
# # print(obj)
# M = None
# camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

cam = cv2.VideoCapture(1)
orb = cv2.ORB_create(nfeatures=10000)
#sift = cv2.xfeatures2d.SIFT_create()
#fgbg = cv2.createBackgroundSubtractorMOG2()
template = cv2.imread("FACE.jpg")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(template_gray, None)
#kpf, desf = sift.detectAndCompute(template_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#bf = cv2.BFMatcher()
#bf = cv2.BFMatcher()
min_matches = 240


while True:
    _, frame = cam.read()
    #frame2 = frame.apply(frame)
    frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # mask = np.zeros(frame.shape[:2],np.uint8)
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    
    # rect = (50,50,450,290)
    # cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # frame = frame*mask2[:,:,np.newaxis]

    
    kp2, des2 = orb.detectAndCompute(frame2, None)
    
    #matches = bf.knnMatch(des1, des2, k=2)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
   
    if len(matches)>min_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = template_gray.shape
        
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)  
        # connect them with lines
        img2 = cv2.polylines(frame, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA) 
       
    
    
    # if M is not None:
    #     try:
    #         projection = projection_matrix(camera_parameters, M)
    #         # frame = render(frame, cont_ini["XYZ"], projection, template_gray, False)
    #         #cv2.imshow("Model AA", frame)
    #     except:
    #         pass
    cv2.imshow("Matching_Live", frame)
        #cv2.imshow("Persp",img2)
    #good = []
    # for m,n in matches:
    #     if m.distance < 0.5*n.distance:
    #         good.append([m])
    # # img3 = cv2.drawMatches(template,kp1,frame,kp2,matches[:10],outImg= None,flags=2) # For only brute force
    # img3 = cv2.drawMatchesKnn(template,kp1,frame,kp2,good,outImg= None,flags=2)
    
    #frame = cv2.drawKeypoints(frame, kp, outImage = None, color=(0,255,0), flags = 0)
    
    
    if cv2.waitKey(33) ==27:   #Press ESC to quit
        break
cam.release()
cv2.destroyAllWindows()

