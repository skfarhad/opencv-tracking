# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 16:44:18 2016

@author: Sk Farhad
"""
import numpy as np
import cv2
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture(os.path.join(dir_path, "tracking_test2.avi"))


''' uncomment this for saving the tracking video
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(os.path.join(dir_path, "tracking_test.avi"), fourcc, 25, (w, h))'''

if __name__ == '__main__':
    
    print("Taking window....")
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)     
     
    prev = None
    cv2.ocl.setUseOpenCL(False)
    
    while(1):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)

        if frame is None:
            break
        
        img=frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is None:            
           prev=frame
           continue
            
        diff = cv2.absdiff(prev, frame)                    
            
        ret,thresh = cv2.threshold(diff,60,255,cv2.THRESH_BINARY);
        blur = cv2.blur(thresh,(10,10))
        thresh = cv2.dilate(blur, None, iterations=10)
        ret,thresh = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY);       
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)                  
        im2, cnts, hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
        if len(cnts)>0:
            for cnt in cnts:
                if cv2.contourArea(cnt)> 100:
                    (x1, y1, w, h) = cv2.boundingRect(cnt)
                    
                    x2 = x1 + w
                    y2 = y1 + h
                    track_window = (x1, y1, (x2-x1), (y2 - y1))
                    x_a,y,w,h = track_window
                    
                    img2 = cv2.rectangle(frame, (x_a,y), (x_a+w,y+h), 255,2)
                    cv2.imshow('frame2',img2) 
    
    cap.release()
    #out.release()
    
    
    
    
    