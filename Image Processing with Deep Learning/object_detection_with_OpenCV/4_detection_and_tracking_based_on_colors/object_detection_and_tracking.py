import cv2
import numpy as np
from collections import deque

# data type to store object center
buffer_size = 16
pts = deque(maxlen = buffer_size)

# color range HSV
lower_band = (46, 98, 0)
upper_band = (114, 255, 255)

# capture
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    
    success, imgOriginal = cap.read()
    
    if success: 
        
        # blur
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) 
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image",hsv)
        
        # create mask for color
        mask = cv2.inRange(hsv, lower_band, upper_band)
        cv2.imshow("mask Image",mask)
        
        # delete noises around the mask
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Mask + Erode + Dilate", mask)
        
        # contour
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            
            c = max(contours, key = cv2.contourArea)
            
            # turn into rectangle
            rect = cv2.minAreaRect(c)
            
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
            
            # draw a dot in the center
            cv2.circle(imgOriginal, center, 5, (120, 120, 0), -1)

            # print information to the screen       
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250, 250, 250), 2)
            
            
        # deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(imgOriginal, pts[i-1], pts[i], (90, 0, 90), 3)
            
        cv2.imshow("Orijinal Detection",imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break
