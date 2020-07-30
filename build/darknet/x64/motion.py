import numpy as np
import cv2 as cv
import sys
import pdb
import time
#pdb.set_trace()

path = "./data/test.mp4"
#path = "./data/aerial.mp4"

cap = cv.VideoCapture(path)

if(cap.isOpened()==False):
    print("Error opening")


#more objects detected and tracked
DetectionPoints = 150


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = DetectionPoints,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15 ,15),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Create some random colors
color = np.random.randint(0,255,(DetectionPoints,3))

loops = 1


ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
        
frame = None
        
while(cap.isOpened()):
    sys.stdout.write("Frame %d     \r" % (loops) )
    sys.stdout.flush()
    
    #every 10 start with new mask
    if(loops%10==0):
        #Every 10 frames readjust
        # Take first frame and find corners in it
        ret, old_frame = ret, frame
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
   

    ret, frame = cap.read()
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    NoneType=type(None)
    
    #make sure something is returned before we try and put it in the frame
    if(not isinstance(p1,NoneType)):
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
    
    
    if(ret==True):
        cv.imshow('Frame',img)
        
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break
    else:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    loops+=1


cap.release()
cv.destroyAllWindows()