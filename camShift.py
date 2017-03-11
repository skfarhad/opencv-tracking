# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 13:57:51 2016

@author: codemen
"""

import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
#cap = cv2.VideoCapture("rtsp://192.168.10.205")
cap = cv2.VideoCapture(os.path.join(dir_path, "tracking_test.avi"))

clicks = 0
points = []

def mouse_position(event,x,y,flags,param):
    
    global clicks 
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if clicks == 0 or clicks == 2 :
            points.append(x)
        else:
            points.append(y)
            
        clicks +=1



if __name__ == '__main__':

    print("Taking window....")
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame',mouse_position)
    
    # take first frame of the video    
    ret, frame = cap.read()       
    while(1):
        if ret==True:
            img=frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Otsu's thresholding after Gaussian filtering
            gray = cv2.GaussianBlur(gray,(5,5),0)
            #print gray.shape 
            #ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            cv2.imshow('frame',img)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if len(points)>3:
                break
        else:
            break
    
    # setup initial location of window
    #r,h,c,w = 250,90,400,125  # simply hardcoded the values    
    #           r   h  c   w 
    #track_window = (c,r,w,h)
    #roi = frame[r:r+h, c:c+w]
    # set up the ROI for tracking
    
    print points
    print("Done!")    
    
    x1 = points[0]
    x2 = points[2]
    y1 = points[1]
    y2 = points[3]
    
    
    
    track_window = (x1, y1, (x2-x1), (y2 - y1))
    
    points = []
    
    
    roi = frame[y1:y2, x1:x2]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    while(1):
        ret ,frame = cap.read()
    
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            
            #x,y,w,h = track_window
            #img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    
            #Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            print pts
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('frame',img2)
            
            
    
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)
    
        else:
            break
    
    cv2.destroyAllWindows()
    cap.release()