import numpy as np
import cv2 as cv

import pdb
#pdb.set_trace()

path = "./data/test.mp4"

cap = cv.VideoCapture(path)

if(cap.isOpened()==False):
    print("Error opening")

while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret==True):
        cv.imshow('Frame',frame)
        
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()