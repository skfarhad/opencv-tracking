# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:33:39 2016

@author: Rifat
"""

import numpy as np
import cv2

import os


dir_path = os.path.dirname(os.path.realpath(__file__))

cv2.ocl.setUseOpenCL(False)

from matplotlib import pyplot as plt


if __name__ == '__main__':
    img1 = cv2.imread(os.path.join(dir_path, "img1_up.jpg"),0)
    img2 = cv2.imread(os.path.join(dir_path, "img2_up.jpg"),0)

    orb = cv2.ORB_create()

    kp = orb.detect(img1,None)
    

    kp, des = orb.compute(img1, kp)
    #plt.imshow(img1),plt.show()
    #gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    imgOut = cv2.drawKeypoints(img1, kp, None, color=(0,255,0), flags=0)
    plt.imshow(imgOut),plt.show()
    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], None, flags=2)
    
    plt.imshow(img3),plt.show()
    
    MIN_MATCH_COUNT = 10
    
    #matches = bf.match(des1,des2)
    
    # store all the good matches as per Lowe's ratio test.

    good = matches[:15]     
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        print matchesMask
        print M
    
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

