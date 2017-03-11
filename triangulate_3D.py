# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:29:49 2016

@author: codemen
"""


import numpy as np
import cv2
import math

import os


dir_path = os.path.dirname(os.path.realpath(__file__))

#cv2.ocl.setUseOpenCL(False)

from matplotlib import pyplot as plt


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


distCoeffs = np.array([-0.01598395733850265, 0.0161002782626507, 0.006006531041189746, 0.003478025937843327, -0.09145387764106144])
camMatrix = np.array([[  3.40435938e+03,   0.00000000e+00,   1.17427106e+03],
    [  0.00000000e+00,   3.40141469e+03,   2.05534346e+03],
    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

def matrixMult(X,Y):
    result = np.array([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X])
    return result
    
def dotMult(X,Y):
    result = 0
    idx = 0
    for x in X:
        result += x*Y[idx]
        idx +=1
    return result
    

        

if __name__ == '__main__':
    img1 = cv2.imread(os.path.join(dir_path, "img1_up.jpg"),0)
    img2 = cv2.imread(os.path.join(dir_path, "img2_up.jpg"),0)

    orb = cv2.xfeatures2d.SIFT_create()

    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    
    '''
    
    #sift = cv2.xfeatures2d.SIFT_create()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #matches = matches[:45]
    
    '''
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    
    #print src_pts

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    matchesMask = mask.ravel().tolist()
    
    print matchesMask
    #print M
    
    #For drawing the boundary    
    '''
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    '''
    draw_params = dict(matchColor = (255,0,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    
    plt.imshow(img3, 'gray'),plt.show()
    
    sizeArray= matchesMask.count(1)

    new_src= np.zeros(shape=(2,sizeArray))
    new_dst= np.zeros(shape=(2,sizeArray))
    indices = [i for i, x in enumerate(matchesMask) if x == 1]
    j=0
    for i in indices:
        #print ("For index ",i)
        #print src_pts[i][0]
        new_src[0][j] = src_pts[i][0][0]
        new_src[1][j] = src_pts[i][0][1]
        new_dst[0][j] = dst_pts[i][0][0]
        new_dst[1][j] = dst_pts[i][0][1]
        j+=1
        
        

            
    #H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(M, camMatrix)
    #A= np.float32(src_pts)
    #print new_src
    #print Rs
    
    RT0=np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    print ("determinants:")
    print np.linalg.det(Rs[0])
    print np.linalg.det(Rs[2])
    
    print("Rs[0] matrix:")
    print Rs[0]
    #print("Ts[0] matrix:")
   # print Ts[0]
    print("Rs[1] matrix:")
    print Rs[1]
    #print("Ts[1] matrix:")
    #print Ts[1]
    print("Rs[2] matrix:")
    print Rs[2]
    #print("Ts[2] matrix:")
    #print Ts[2]
    print("Rs[3] matrix:")
    print Rs[3]
   # print("Ts[3] matrix:")
    #print Ts[3]    
    
    '''
    RT1 = np.hstack((Rs[0],Ts[0]))
    print RT1

    RT2 = np.hstack((Rs[1],Ts[1]))
    print RT2
    
    RT3 = np.hstack((Rs[2],Ts[2]))
    print RT3
    RT4 = np.hstack((Rs[3],Ts[3]))
    print RT4
    
    Tpoints1 = cv2.triangulatePoints(RT0, RT1 , new_src, new_dst)  
    print("3d points RT1:")
    print Tpoints1
    
    Tpoints2 = cv2.triangulatePoints(RT0, RT2 , new_src, new_dst)  
    print("3d points RT2:")
    print Tpoints2 
    
    Tpoints3 = cv2.triangulatePoints(RT0, RT3 , new_src, new_dst)  
    print("3d points RT3:")
    print Tpoints3 
    
    Tpoints4 = cv2.triangulatePoints(RT0, RT4 , new_src, new_dst)  
    print("3d points RT4:")
    print Tpoints4 
    
    print Ns[0]
    print Ns[2]
    '''
    
    print("R1.N1:")
    print dotMult(Rs[0][2],Ns[0])
    print("R3.N3:")
    print dotMult(Rs[2][2],Ns[2])  
    theta = rotationMatrixToEulerAngles(Rs[0])
    
    A = 180/3.1416
    
    tx = A*theta[0]
    ty = A*theta[1]
    tz = A*theta[2]
    
    print tx
    print ty 
    print tz
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    

