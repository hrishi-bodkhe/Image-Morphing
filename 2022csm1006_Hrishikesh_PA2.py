#!/usr/bin/env python
# coding: utf-8

# In[1135]:


#Name:- Hrishikesh Bodkhe
#Enrolment No.:- 2022CSM1006


# Some libraries and modules are used which are needed to be installed first using the following commands:- 
# 
# A. Installing dilb library:-
# 1. conda install -c conda-forge dlib
# 2. pip install --upgrade dlib
# 
# B. Installing the moviepy module to create gif:-
# 1. pip install moviepy

# Both parts of the assignment are done in the same notebook. Functions are defined in the beginning. A markdown indicates from where PART A & PART B starts.
# 
# The folder consists of the images used:- "pic1.jpg" & "pic2.jpg". The final output is stored as "morphedImageA.gif" & "morphedImageB.gif" in the same folder

# Pic 1 Reference:- https://raw.githubusercontent.com/KubricIO/face-morphing/master/demos/ben.jpg
# 
# Pic 2 Reference:- https://raw.githubusercontent.com/KubricIO/face-morphing/master/demos/morgan.jpg

# "shape_predictor_68_face_landmarks.dat" - This file is needed to detect 68 tie points in image using dlib library, which is included in the folder. The same file can be found here:- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# In[1136]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import dlib
from moviepy.editor import ImageSequenceClip


# In[1137]:


def cvShow(img):
    b,g,r = cv.split(img)
    opimg = cv.merge([r, g, b])
    plt.axis('off')
    plt.imshow(opimg)


# In[1138]:


def imgShape(img, i):
    print('Image', i, 'Shape:', img.shape)


# In[1139]:


def giveTiePoints(tiePoints, features):
    for point in features.parts():
        X = point.x
        Y = point.y
        
        tiePoints.append(list((X, Y)))
    
    return np.array(tiePoints)


# In[1140]:


def showTiePointsInImage(img, tiePoints):
    for point in tiePoints:
        x = point[0]
        y = point[1]
        
        cv.circle(img, (x, y), 4, (0, 0, 255), -1)
    
    cvShow(img)


# In[1141]:


def triangulate(img, tiePoints, imgWithTriangles):
    rect = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv.Subdiv2D(rect)
    
    for point in tiePoints:
        x, y = int(point[0]), int(point[1])
        subdiv.insert((x, y))
    
    triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)
#     print(triangles)
    
    for t in triangles:
        x1, y1 = t[0], t[1]
        x2, y2 = t[2], t[3]
        x3, y3 = t[4], t[5]
    
        cv.line(imgWithTriangles, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv.line(imgWithTriangles, (x2, y2), (x3, y3), (0, 255, 255), 1)
        cv.line(imgWithTriangles, (x3, y3), (x1, y1), (0, 255, 255), 1)
    
    return imgWithTriangles, triangles


# In[1142]:


def findIndexOfPoints(points, target):
    for idx, point in enumerate(points):
        if np.all(point == target):
            return idx
    return -1


# In[1143]:


def triangulateUsingImage(triangles1, tiePoints1, tiePoints2):
    triangles2 = []
    
    for triangle in triangles1:
        point1 = [triangle[0], triangle[1]]
        point2 = [triangle[2], triangle[3]]
        point3 = [triangle[4], triangle[5]]
        
        idx1 = findIndexOfPoints(tiePoints1, point1)
        idx2 = findIndexOfPoints(tiePoints1, point2)
        idx3 = findIndexOfPoints(tiePoints1, point3)
        
        newPoint1 = [tiePoints2[idx1][0], tiePoints2[idx1][1]]
        newPoint2 = [tiePoints2[idx2][0], tiePoints2[idx2][1]]
        newPoint3 = [tiePoints2[idx3][0], tiePoints2[idx3][1]]
        
        newTriangle = [newPoint1[0], newPoint1[1], newPoint2[0], newPoint2[1], newPoint3[0], newPoint3[1]]
        triangles2.append(newTriangle)
    
    return np.array(triangles2, dtype=np.int32)


# In[1144]:


def showTrianglesInImage(img, triangles):
    for t in triangles:
        x1, y1 = t[0], t[1]
        x2, y2 = t[2], t[3]
        x3, y3 = t[4], t[5]
    
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv.line(img, (x2, y2), (x3, y3), (0, 255, 255), 1)
        cv.line(img, (x3, y3), (x1, y1), (0, 255, 255), 1)
    
    cvShow(img)


# In[1145]:


def extractRectangle(img, rectangle):
    img = np.float32(img)
    sx, sy = rectangle[0], rectangle[1]
    ex, ey = sx + rectangle[2], sy + rectangle[3]
    
    return img[sy:ey, sx:ex]


# # ==========================================================

# # PART A

# In[1146]:


img1 = cv.imread('pic1.jpg')
img2 = cv.imread('pic2.jpg')


# In[1147]:


imgShape(img1, 1)
imgShape(img2, 2)


# ### Reading tie points from file tiepoints.txt

# In[1148]:


tiePoints1 = []
tiePoints2 = []

with open('tiePoints.txt', 'r') as file:
    T = int(file.readline())
    
    for i in range(T):
        points = list(map(int, file.readline().split()))
        tiePoints1.append([points[0], points[1]])
        tiePoints2.append([points[2], points[3]])

file.close()

tiePoints1 = np.array(tiePoints1, dtype=np.int32)
tiePoints2 = np.array(tiePoints2, dtype=np.int32)


# In[1149]:


imgWithTiePoints1 = img1.copy()
showTiePointsInImage(imgWithTiePoints1, tiePoints1)


# In[1150]:


imgWithTiePoints2 = img2.copy()
showTiePointsInImage(imgWithTiePoints2, tiePoints2)


# ### Delaunay Triangulation

# In[1151]:


imgWithTriangles1 = img1.copy()
imgWithTriangles1, triangles1 = triangulate(img1, tiePoints1, imgWithTriangles1)
cvShow(imgWithTriangles1)


# In[1152]:


triangles2 = triangulateUsingImage(triangles1, tiePoints1, tiePoints2)
imgWithTriangles2 = img2.copy()
showTrianglesInImage(imgWithTriangles2, triangles2)


# ### Morphing

# In[1153]:


plt.figure(figsize=(20, 20))
index = 1
morphedImages1 = []
for _lambda in np.arange(0, 1, 0.1):
    
    #Calculating the tie points in morphed image
    morphedTiePoints = []
    n = len(tiePoints1)
    
    for k in range(n):
        xi, yi = tiePoints1[k][0], tiePoints1[k][1]
        xj, yj = tiePoints2[k][0], tiePoints2[k][1]
    
        xm = int((1 - _lambda) * xi + _lambda * xj)
        ym = int((1 - _lambda) * yi + _lambda * yj)
    
        morphedTiePoints.append((xm, ym))

    morphedTiePoints = np.array(morphedTiePoints, dtype=np.int32)
    
    #Constructing a black image to store the result 
    morphedImage = np.zeros(img1.shape, dtype=img.dtype)
    
    #Triangulating in morphed image
    morphedTriangles = triangulateUsingImage(triangles1, tiePoints1, morphedTiePoints)
    
    for i in range(len(morphedTriangles)):
        #from image 1
        triangle1 = [(triangles1[i][0], triangles1[i][1]), (triangles1[i][2], triangles1[i][3]), (triangles1[i][4], triangles1[i][5])]
    
        #from image 2
        triangle2 = [(triangles2[i][0], triangles2[i][1]), (triangles2[i][2], triangles2[i][3]), (triangles2[i][4], triangles2[i][5])]
    
        #from morphed image
        triangleM = [(morphedTriangles[i][0], morphedTriangles[i][1]), (morphedTriangles[i][2], morphedTriangles[i][3]), (morphedTriangles[i][4], morphedTriangles[i][5])]
    
    
        #Finding bounding rectangle for each triangle
        br1 = cv.boundingRect(np.float32([triangle1]))
        br2 = cv.boundingRect(np.float32([triangle2]))
        brM = cv.boundingRect(np.float32([triangleM]))
    
        #extracting top-left corner coordinates of bounding rectangle
        rx1, ry1 = br1[0], br1[1]
        rx2, ry2 = br2[0], br2[1]
        rxM, ryM = brM[0], brM[1]
    
        for j in range(3):
            #extracting triangle coordinates
            x1, y1 = triangle1[j][0], triangle1[j][1]
            x2, y2 = triangle2[j][0], triangle2[j][1]
            xM, yM = triangleM[j][0], triangleM[j][1]
        
            #normalizing the triangle points
            new_x1, new_y1 = x1 - rx1, y1 - ry1
            new_x2, new_y2 = x2 - rx2, y2 - ry2
            new_xM, new_yM = xM -rxM, yM - ryM
        
            #updating the triangle points
            triangle1[j] = (new_x1, new_y1)
            triangle2[j] = (new_x2, new_y2)
            triangleM[j] = (new_xM, new_yM)
    
        #creating the mask
        mask = np.zeros((brM[3], brM[2], 3), dtype=np.float32)
        cv.fillConvexPoly(mask, np.int32(triangleM), (1.0, 1.0, 1.0), 16, 0)
    
        #extracting the rectangular region from image
        imgRect1 = extractRectangle(img1, br1)
        imgRect2 = extractRectangle(img2, br2)
    
        #finding the transformation matrix
        transfMat1 = cv.getAffineTransform(np.float32(triangle1), np.float32(triangleM))
        transfMat2 = cv.getAffineTransform(np.float32(triangle2), np.float32(triangleM))
    
        #applying the transformation matrix to perform the affine transformation
        transformation1 = cv.warpAffine(imgRect1, transfMat1, (brM[2], brM[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        transformation2 = cv.warpAffine(imgRect2, transfMat2, (brM[2], brM[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    
        #blending the rectangles
        morphedRect = (1 - _lambda) * transformation1 + _lambda * transformation2
    
        #updating the output / morphed image with new rectangle
        sx, sy = brM[0], brM[1]
        ex, ey = sx + brM[2], sy + brM[3]
        morphedImage[sy:ey, sx:ex] = morphedImage[sy:ey, sx:ex] * (1 - mask) + morphedRect * mask
    
    b,g,r = cv.split(morphedImage)
    opimg = cv.merge([r, g, b])
    morphedImages1.append(opimg)
    plt.subplot(3, 4, index)
    index = index + 1
    cvShow(morphedImage)


# In[1154]:


clip1 = ImageSequenceClip(morphedImages1, fps=5000)
clip1.write_gif('morphedImageA.gif')


# # ============================================================

# # PART B

# In[1155]:


img1 = cv.imread('pic1.jpg')
img2 = cv.imread('pic2.jpg')


# In[1156]:


img = np.hstack((img1, img2))
cvShow(img)


# In[1157]:


imgShape(img1, 1)
imgShape(img2, 2)


# ### Finding Tie Points

# In[1158]:


detector = dlib.get_frontal_face_detector()


# In[1159]:


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# In[1160]:


face1 = detector(img1)[0]
features1 = predictor(img1, face1)


# In[1161]:


face2 = detector(img2)[0]
features2 = predictor(img2, face2)


# In[1162]:


tiePoints1 = []
tiePoints2 = []

tiePoints1 = np.array(giveTiePoints(tiePoints1, features1), dtype=np.int32)
tiePoints2 = np.array(giveTiePoints(tiePoints2, features2), dtype=np.int32)


# In[1163]:


additionalTiePoints1 = np.array([[0, 0], [93, 0], [197, 0], [303, 0], [399, 0], [72, 24], [51, 62], [48, 101], [54, 140],
                                 [326, 40], [341, 106], [334, 160], [111, 175], [111, 129], [133, 91], [187, 68], [267, 60],
                                [318, 79], [329, 125], [326, 163], [77, 175], [56, 150], [51, 194], [65, 227], [84, 252],
                                [333, 162], [327, 212], [0, 214], [0, 368], [0, 459], [399, 230], [399, 459], [236, 422],
                                [77, 284], [20, 354], [265, 384], [350, 433], [236, 425], [98, 459], [174, 459], [240, 459],
                                [307, 459], [376, 459]], dtype=np.int32)


# In[1164]:


additionalTiePoints2 = np.array([[0, 0], [144, 0], [200, 0], [243, 0], [399, 0], [90, 30], [73, 55], [58, 112], [67, 160],
                                [312, 40], [343, 138], [316, 210], [98, 190], [122, 127], [150, 66], [205, 69], [280, 83],
                                [296, 117], [305, 169], [310, 200], [77, 173], [66, 156], [67, 199], [75, 229], [89, 250],
                                [314, 209], [307, 246], [0, 214], [0, 336], [0, 459], [399, 230], [399, 459], [188, 405], 
                                [91, 289], [39, 326], [256, 362], [383, 420], [198, 435], [98, 459], [138, 459], [220, 459], 
                                [289, 459], [352, 459]])


# In[1165]:


tiePoints1 = np.concatenate((tiePoints1, additionalTiePoints1), axis=0)


# In[1166]:


tiePoints2 = np.concatenate((tiePoints2, additionalTiePoints2), axis=0)


# In[1167]:


imgWithTiePoints1 = img1.copy()
showTiePointsInImage(imgWithTiePoints1, tiePoints1)


# In[1168]:


imgWithTiePoints2 = img2.copy()
showTiePointsInImage(imgWithTiePoints2, tiePoints2)


# ### Delaunay Triangulation

# In[1169]:


imgWithTriangles1 = img1.copy()
imgWithTriangles1, triangles1 = triangulate(img1, tiePoints1, imgWithTriangles1)
cvShow(imgWithTriangles1)


# In[1170]:


triangles2 = triangulateUsingImage(triangles1, tiePoints1, tiePoints2)
imgWithTriangles2 = img2.copy()
showTrianglesInImage(imgWithTriangles2, triangles2)


# ### Morphing

# In[1171]:


plt.figure(figsize=(20, 20))
index = 1
morphedImages2 = []
for _lambda in np.arange(0, 1, 0.1):
    
    #Calculating the tie points in morphed image
    morphedTiePoints = []
    n = len(tiePoints1)
    
    for k in range(n):
        xi, yi = tiePoints1[k][0], tiePoints1[k][1]
        xj, yj = tiePoints2[k][0], tiePoints2[k][1]
    
        xm = int((1 - _lambda) * xi + _lambda * xj)
        ym = int((1 - _lambda) * yi + _lambda * yj)
    
        morphedTiePoints.append((xm, ym))

    morphedTiePoints = np.array(morphedTiePoints, dtype=np.int32)
    
    #Constructing a black image to store the result 
    morphedImage = np.zeros(img1.shape, dtype=img.dtype)
    
    #Triangulating in morphed image
    morphedTriangles = triangulateUsingImage(triangles1, tiePoints1, morphedTiePoints)
    
    for i in range(len(morphedTriangles)):
        #from image 1
        triangle1 = [(triangles1[i][0], triangles1[i][1]), (triangles1[i][2], triangles1[i][3]), (triangles1[i][4], triangles1[i][5])]
    
        #from image 2
        triangle2 = [(triangles2[i][0], triangles2[i][1]), (triangles2[i][2], triangles2[i][3]), (triangles2[i][4], triangles2[i][5])]
    
        #from morphed image
        triangleM = [(morphedTriangles[i][0], morphedTriangles[i][1]), (morphedTriangles[i][2], morphedTriangles[i][3]), (morphedTriangles[i][4], morphedTriangles[i][5])]
    
        #Finding bounding rectangle for each triangle
        br1 = cv.boundingRect(np.float32([triangle1]))
        br2 = cv.boundingRect(np.float32([triangle2]))
        brM = cv.boundingRect(np.float32([triangleM]))
        
        #extracting top-left corner coordinates of bounding rectangle
        rx1, ry1 = br1[0], br1[1]
        rx2, ry2 = br2[0], br2[1]
        rxM, ryM = brM[0], brM[1]
    
        for j in range(3):
            #extracting triangle coordinates
            x1, y1 = triangle1[j][0], triangle1[j][1]
            x2, y2 = triangle2[j][0], triangle2[j][1]
            xM, yM = triangleM[j][0], triangleM[j][1]
        
            #normalizing the triangle points
            new_x1, new_y1 = x1 - rx1, y1 - ry1
            new_x2, new_y2 = x2 - rx2, y2 - ry2
            new_xM, new_yM = xM -rxM, yM - ryM
        
            #updating the triangle points
            triangle1[j] = (new_x1, new_y1)
            triangle2[j] = (new_x2, new_y2)
            triangleM[j] = (new_xM, new_yM)
    
        #creating the mask
        mask = np.zeros((brM[3], brM[2], 3), dtype=np.float32)
        cv.fillConvexPoly(mask, np.int32(triangleM), (1.0, 1.0, 1.0), 16, 0)
    
        #extracting the rectangular region from image
        imgRect1 = extractRectangle(img1, br1)
        imgRect2 = extractRectangle(img2, br2)
    
        #finding the transformation matrix
        transfMat1 = cv.getAffineTransform(np.float32(triangle1), np.float32(triangleM))
        transfMat2 = cv.getAffineTransform(np.float32(triangle2), np.float32(triangleM))
    
        #applying the transformation matrix to perform the affine transformation
        transformation1 = cv.warpAffine(imgRect1, transfMat1, (brM[2], brM[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        transformation2 = cv.warpAffine(imgRect2, transfMat2, (brM[2], brM[3]), None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    
        #blending the rectangles
        morphedRect = (1 - _lambda) * transformation1 + _lambda * transformation2
    
        #updating the output / morphed image with new rectangle
        sx, sy = brM[0], brM[1]
        ex, ey = sx + brM[2], sy + brM[3]
        morphedImage[sy:ey, sx:ex] = morphedImage[sy:ey, sx:ex] * (1 - mask) + morphedRect * mask
    
    b,g,r = cv.split(morphedImage)
    opimg = cv.merge([r, g, b])
    morphedImages2.append(opimg)
    plt.subplot(3, 4, index)
    index = index + 1
    cvShow(morphedImage)


# In[1172]:


clip2 = ImageSequenceClip(morphedImages1, fps=5000)
clip2.write_gif('morphedImageB.gif')

