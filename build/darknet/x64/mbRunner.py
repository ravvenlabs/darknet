import numpy as np
import cv2
import sys
import pdb
import time
from datetime import datetime
from yolo_display_utils import *
from mv_utils import *
from macroBlock import *

#pdb.set_trace()
path = "./data/test.mp4"
#path = "./data/aerial.mp4"

cap = cv2.VideoCapture(path)
cap.set(3, 416)
cap.set(4, 416)
if(cap.isOpened()==False):
    print("Error opening")
  
MV_YOLO_ASSOCIATION_BUFFER_X = 5
MV_YOLO_ASSOCIATION_BUFFER_Y = 5
  
###Insert at top of file###


pixelH = 416
pixelV = 416

pixels=pixelH*pixelV
MB_SIZE = 16

numBlocks = int(pixels / (MB_SIZE*MB_SIZE))
numBlocksHorz = int(pixelH/MB_SIZE)
numBlocksVert = int(pixelV/MB_SIZE)

macroBlockListPrev = []
macroBlockListCur = []
macroBlockAbsLocation = []
macroBlockAbsLocationPre = []
mv_boxes = []

xMotionVector = [None]* numBlocks
yMotionVector = [None]*numBlocks

#macroBlockAbsIDXs = []
searchWindow = 10
searchWindow = 7
frame_gray_prev =None
frame_gray = None

MB_PARAM = (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH)
#MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation)
###



#A list of detections 
detectionsCanned = [(b'car', 0.9875175356864929, (132.9082489013672, 252.57867431640625, 19.92416000366211, 27.44229507446289)), (b'car', 0.9844262599945068, (187.90623474121094, 186.30271911621094, 9.99842643737793, 12.396671295166016)), (b'car', 0.9786965847015381, (291.46466064453125, 269.5498962402344, 24.154693603515625, 34.94427490234375)), (b'car', 0.9781897068023682, (248.66781616210938, 249.05857849121094, 19.71929359436035, 35.382442474365234)), (b'car', 0.9753296971321106, (166.82289123535156, 197.5367431640625, 13.600704193115234, 16.434371948242188)), (b'car', 0.9693384766578674, (180.45913696289062, 220.43231201171875, 14.002452850341797, 20.855995178222656)), (b'car', 0.9488425850868225, (149.05673217773438, 195.62550354003906, 11.167325019836426, 11.12906265258789)), (b'car', 0.9218205809593201, (354.57940673828125, 310.01837158203125, 38.63389205932617, 51.30183029174805)), (b'car', 0.8971863985061646, (225.61891174316406, 181.2677459716797, 8.977764129638672, 12.375943183898926)), (b'truck', 0.8725090622901917, (253.3011474609375, 174.546142578125, 17.84792709350586, 33.19464874267578)), (b'car', 0.8053892254829407, (219.83656311035156, 163.845947265625, 8.55410385131836, 10.328858375549316)), (b'car', 0.6540932059288025, (81.79322052001953, 270.52166748046875, 29.485111236572266, 33.689449310302734)), (b'car', 0.6494554877281189, (119.26595306396484, 401.3941650390625, 57.44500732421875, 31.289005279541016)), (b'car', 0.6438901424407959, (180.33819580078125, 171.95281982421875, 9.02131462097168, 9.384852409362793)), (b'car', 0.6157169342041016, (164.00958251953125, 174.623046875, 10.016670227050781, 10.10055160522461)), (b'car', 0.5976225733757019, (171.83770751953125, 185.80467224121094, 9.461698532104492, 11.111896514892578)), (b'car', 0.5843924880027771, (237.10079956054688, 163.36111450195312, 7.352409839630127, 7.880095958709717)), (b'truck', 0.5014382600784302, (81.72644805908203, 269.6203308105469, 30.076330184936523, 42.456581115722656)), (b'car', 0.437776654958725, (230.6274871826172, 154.78915405273438, 5.732120513916016, 6.44113302230835)), (b'bus', 0.4131737947463989, (254.37860107421875, 176.56822204589844, 16.1439208984375, 31.278751373291016))]
detectionsCannedOrig = detectionsCanned.copy()
#pdb.set_trace()
loops = 1

#### Macroblock get

#def InitMacroBlockGlobals():
#    global macroBlockListPrev
#    global macroBlockListCur
#    global macroBlockAbsLocationPre
#    
#    return None

start_time = datetime.now()

ret, old_frame = cap.read()

old_frame = cv2.resize(old_frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
        
frame = None

breakAt = 501

ret, frame = cap.read()  
frame = cv2.resize(frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
now =  datetime.now()

MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector)

#For current frame
while(cap.isOpened()):

    

    sys.stdout.write("Frame %d     \r" % (loops) )
    sys.stdout.flush()
    
    #LOOP timing
    prev = now
    now = datetime.now()
    print((now-prev).total_seconds(), " Seconds elapsed!")
    #
   
    #READ FRAME
    ret, frame = cap.read()
 
    #STORE PREV
    frame_gray_prev = frame_gray.copy()
    
    #RESIZE NEW FRAME
    frame = cv2.resize(frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)
    #MAKE GRAYSCALE
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
    
    #################Macroblock function pass in both frames
    ######################GETBLOCKS##########################################
    
    
    xMotionVector, yMotionVector, MB_LISTS = GetMacroBlockMotionVectors(MB_PARAM, MB_LISTS, frame_gray, frame_gray_prev)

    
    #pdb.set_trace()
    
    mv_boxes, MB_LISTS = UpdateMvBoxesMacroBlock(mv_boxes, MB_PARAM, MB_LISTS, detectionsCanned, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y)
    
        
    if(not len(mv_boxes)==0 ):  
        
        ##Draw results
        frame = cvDrawBoxes(detectionsCannedOrig,frame, COLOR_green)
        frame = cvDrawBoxes(mv_boxes,frame, COLOR_red)
        
        detectionsCanned.clear()
        detectionsCanned = mv_boxes.copy()
        mv_boxes.clear()
            
    if(ret==True):
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else:
        break
    
 
    
    loops+=1
    if(loops==breakAt):
        break

now = datetime.now()
print((start_time).total_seconds(), " Seconds elapsed total!")

print((now).total_seconds(), " Seconds elapsed total!")

cap.release()
cv2.destroyAllWindows()