from ctypes import *
import math
import random
import os
import cv2

import numpy as np
import time
import sys
#import darknet
import math
import pdb
import matplotlib.pyplot as plt

from yolo_display_utils import *
from mv_utils import *
from copy import deepcopy

from macroBlock import *

from general_utils import *


#openCV static info
path = ".\\data\\test.mp4"
#path = "./data/simpler_trim.mp4"
#path = "./data/two_min_alps_traffic.mp4"

#path = ".\\data\\aerial_Trim.mp4"

####################################################################
#Create Some lists that will store detections
AllDetections = []
AllDetectionsDelayed = []

AllMVBoxes = []
AllMVBoxesDelayed = []

AllMatchedBoxes_MV_YOLO = []
AllMatchedBoxes_MV_YOLODelayed = []

AllMatchedBoxes_YOLO_to_GT = []
AllMatchedBoxes_MV_to_GT = []

All_OTB_GT = []


FrameCumulativeDriftMVYOLO = []
FrameCumulativeDriftYOLO = []
FrameCumulativeDriftMV = []
#USE OTB DATA


#########################################
####MACROBLOCK CODE/PARAMS
#########################################
USE_MB_MOTION = False
if(USE_MB_MOTION):

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


    MB_PARAM = (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH)
    MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector)

#################

DETECT_DELAY=False

#Using OTB will cuase the program to read in OTB data which is a CV benchmark set
USE_OTB = True
PLOT_AND_COMPARE_CENTERS = False
        
if(USE_OTB):
    OTB_GT_FIX_TIME = False
    
    
    OTB_GT_FIX_TIME = True
    path = ".\data\OTB_data\stationary\Walking2\otb_Walking2.avi"
    otb_gt_file = ".\data\OTB_data\stationary\Walking2\groundtruth_rect.txt"
      
    OTB_GT_FIX_TIME = False
      
    #path = ".\data\OTB_data\sway\CarScale_from_paper\CarScale_from_paper.avi"
    #otb_gt_file = ".\data\OTB_data\sway\CarScale_from_paper\groundtruth_rect.txt"
      
    #path = ".\data\OTB_data\sway\David3\David3.avi"
    #otb_gt_file = ".\data\OTB_data\sway\David3\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\sway\Human4\Human4.avi"
    #otb_gt_file = ".\data\OTB_data\sway\Human4\groundtruth_rect.2.txt"
    
    #path = ".\data\OTB_data\sway\Human5\Human5.avi"
    #otb_gt_file = ".\data\OTB_data\sway\Human5\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\sway\Jogging\Jogging.avi"
    #otb_gt_file = ".\data\OTB_data\sway\Jogging\groundtruth_rect.1.txt"
    
    #path = ".\data\OTB_data\sway\Woman\Woman.avi"
    #otb_gt_file = ".\data\OTB_data\sway\Woman\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\sway\Couple\Couple.avi"
    #otb_gt_file = ".\data\OTB_data\sway\Couple\groundtruth_rect.txt"
    
    
      
    #OTB_GT_FIX_TIME = False
    
    #OTB_GT_FIX_TIME = False
    
    #path = ".\data\OTB_data\stationary\Walking\otb_Walking.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Walking\groundtruth_rect.txt"
    
    
    #path = ".\data\OTB_data\stationary\Crossing\otb_crossing.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Crossing\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\stationary\Subway\otb_Subway.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Subway\groundtruth_rect.txt"
    
    #TODO fix for CSV
    #path = ".\data\OTB_data\stationary\Crowds\otb_Crowds.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Crowds\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\stationary\Dancer2\otb_Dancer2.avi"
    #TODO fix for CSV

    #otb_gt_file = ".\data\OTB_data\stationary\Dancer2\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\stationary\Man\otb_Man.avi"
    #TODO fix for CSV
    #otb_gt_file = ".\data\OTB_data\stationary\Man\groundtruth_rect.txt"

    OTB_DETECT_PEOPLE_ONLY = True

    #WHEN USING! make sure the path is to an otb data folder as well

    #plot otb ground truth
    SHOW_OTB_GT = False
    
    
    ####################################################################
    #calulate and store metrics for precision and success plots
    CALC_OTB_METRICS_STORE = True
    OTB_OUT_PRECISION_SUFFIX = "otb_prec_plot.txt"
    OTB_OUT_SUCCESS_SUFFIX = "otb_succes_plot.txt"
    
    #0 to 100
    SUCCESS_THRESH = 0
    
    str_suc = str(SUCCESS_THRESH)

    #0 to 50
    PIXEL_DIST_THRESH = 0
    #PIXEL_DIST_THRESH = pix_dist_thing
    
    #Success is %of detections where IOU is over threshold from 0 to 100%
    #precision is % of detections where centroid is within *threshold* number of pixels of gt

    OTB_FOLDER= ".\OTB_OUT\\"

    SUCCESS_FILE = OTB_FOLDER + str(SUCCESS_THRESH)+OTB_OUT_SUCCESS_SUFFIX
    PIXEL_DIST_FILE = OTB_FOLDER + str(PIXEL_DIST_THRESH)+OTB_OUT_PRECISION_SUFFIX
    
    success_fptr = open(SUCCESS_FILE, "w")
    prec_fptr = open(PIXEL_DIST_FILE, "w")
    
    ####################################################################
    
else:
    SHOW_OTB_GT = False
    OTB_GT_FIX_TIME = False



#Frames till next optical flow point refresh
OF_DET_SKIP = 4

#Force darknet to predict and redraw every frame
ALWAYS_REDRAW_Redetect = True

#frames till next yolo call 
YOLO_DET_SKIP =10

#PRint and plot framerate
PRINT_FRAMERATE = False

#Rectangles for box movement and crosshairs on MV in said rectangle (And print locations of each)
DEBUG_OBJECTS = False

#MV inside of rectangle buffer for outside of rectangle [units of pixels]
MV_RECT_BUFFER_VERT = 0
MV_RECT_BUFFER_HORZ = 0

#Yolo accuracy required to make a bbox
YOLO_DET_THRESH = .35

#darknet hierarchical threshold - how specific the detections should be with label
HI_THRESH=.5

#non max suppression threshold. Stops multiple boxes from being placed in the same spot
NMS=.30

#Insert a delay from one frame to the next
SLOW_MODE =False

#with VOC
#YOLO_DET_THRESH = .10

#When matching yolo detections to existing MV bboxes, this is the overlap buffer they can be off by and still "intersect"
MV_YOLO_ASSOCIATION_BUFFER_X = 5
MV_YOLO_ASSOCIATION_BUFFER_Y = 5

#put circles on most recent O.F. point
CV_CIRCLE_ON = False
#put lines on for O.F.
CV_LINES_ON = True

#more objects detected and tracked with O.F.
DetectionPoints = 250

#draw yolo bboxes (every frame green)
drawYOLO = True

#Draw motion-vector-propelled boxes
Draw_MV_BOXES = True

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = DetectionPoints,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
                                 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15 ,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

###MODELS##

#yolov4
#configPath = "./cfg/yolov4.cfg"
#weightPath = "./weights/yolov4.weights"
#metaPath = "./cfg/coco.data"

#yolov3
configPath = "./cfg/yolov3.cfg"
weightPath = "./weights/yolov3.weights"
metaPath = "./cfg/coco.data"

#yolov2 tiny
#configPath = "./cfg/yolov2-tiny.cfg"
#weightPath = "./weights/yolov2-tiny.weights"
#metaPath = "./cfg/coco.data"

#yolov3 tiny
#configPath = "./cfg/yolov3-tiny.cfg"
#weightPath = "./weights/yolov3-tiny.weights"
#metaPath = "./cfg/coco.data"


#-->VOC is horrible currently
#configPath = "./cfg/yolov2-tiny-voc.cfg"
#weightPath = "./weights/yolov2-tiny-voc.weights"
#metaPath = "./cfg/voc.data"

#####################################################################


frame = 0
# ####################

netMain = None
metaMain = None
altNames = None



#Main loop
def YOLO():

    #init some variables for the loop
    frame=0
    frame_read= None
    MVBoxes = None
    dbgFrame = None
    detections = None
    maxx=0
    maxy=0
    NoneType=type(None)
    
    cumulitiveDrift = 0
    
    driftArr = []
    frameRateArr = []
    loopsArr = []
    
    #otb gt list
    otblist = []
    
    #Run for x frames of video. This is for plot consistency in videos
    breakAt = 500
    
    #Buffered display
    DisplayBuffer = [None]*YOLO_DET_SKIP
    DetectionsBuffer = [None]*YOLO_DET_SKIP
    DetectionsEveryFrameBuffer = [None]*YOLO_DET_SKIP
    LastDetection = None
    NextDet = None
    
    MVBuffer = [None]*YOLO_DET_SKIP
    SanityBuffer = [None]*YOLO_DET_SKIP

    origYolo = None
    detection_delayed = None
    mvbox_delayed = None
    frame_gray=[]
    addedToFrame = False


    global MB_LISTS, MB_PARAM
    global metaMain, netMain, altNames

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(path)
    
    #Record factors to scale by
    if(USE_OTB):
        otb_width  = cap.get(3) 
        otb_height = cap.get(4) 
        #c
        otb_x_scale = ( darknet.network_width(netMain) / otb_width)
        otb_y_scale = ( darknet.network_height(netMain) / otb_height )
    
    cap.set(3, darknet.network_width(netMain))
    cap.set(4, darknet.network_height(netMain))
    #cap.set(3, 1280)
    #cap.set(4, 720)

    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)


    #get frame
    ret, frame_read = cap.read()
    if(USE_OTB):
        otb_file = open(otb_gt_file)
        otb_file.readline()
        
    #Get rid of the camera timer that messes up optical flow
    if(OTB_GT_FIX_TIME):
        pt1 = (0,0)
        pt2 = (220,15)
        frame_read = cv2.rectangle(frame_read, pt1, pt2, (0,0,0), -10)
        
    #resize to darknet preference
    frame_read = cv2.resize(frame_read, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
                                    
    #Get the first frame in gray
    old_gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
    
    #Use opencv to gather the points to track with lk
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame_read)
    dotMask = np.zeros_like(frame_read)
    
    now = datetime.now()
    
    ##########################   
    ## FRAME READ MAIN LOOP ##
    ##########################

    #While the video is open
    while cap.isOpened():   
#        
        
        prev = now
        now = datetime.now()
        print((now-prev).total_seconds(), " Seconds elapsed!")
        prev_time = time.time()
        
        loopsArr.append(frame)
        
        if(SLOW_MODE):
            time.sleep(0.05)
        
        frameIndex=frame
        frame=frame+1
        
        #every 10 start with new mask
        if(frame%OF_DET_SKIP==0):
            #Every 10 frames readjust
            # Take first frame and find corners in it
            ret, old_frame = ret, frame_read
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            
            #get new tracking points
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
        
        #Read the next frame
        ret, frame_read = cap.read()
        
        #Video failed to return another frame
        if(not ret):
            break
        
        #Get rid of the camera timer that messes up optical flow
        if(OTB_GT_FIX_TIME):
            pt1 = (0,0)
            pt2 = (220,15)
            frame_read = cv2.rectangle(frame_read, pt1, pt2, (0,0,0), -10)
            
        if(frame > breakAt and (PRINT_FRAMERATE or PLOT_AND_COMPARE_CENTERS)):
            break
    
        #resize the frame for darknet
        frame_read = cv2.resize(frame_read, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        
        ###############################MVs#########################
        ##optical flow code#
        
        #Gray frame previous and current
        frame_gray_prev = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        #make sure something is returned before we try and put it in the frame
        if(not isinstance(p1,NoneType)):
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            #This loop draws open cv features
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                if(CV_LINES_ON):
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                if(CV_CIRCLE_ON):
                    frame_rgb = cv2.circle(frame_rgb,(a,b),5,color[i].tolist(),-1)

        #add mask to image
        image = cv2.add(frame_rgb,mask)
       
        #cv2.imshow('Demo', image)
        #################################################
        ##Darknet Detection#
        
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        
        #This block will get detections every frame and draw if needed
        if(ALWAYS_REDRAW_Redetect or PLOT_AND_COMPARE_CENTERS):
            detectionsEveryFrame = darknet.detect_image(netMain, metaMain, darknet_image, thresh=YOLO_DET_THRESH, hier_thresh=HI_THRESH, nms=NMS)
            COLOR = COLOR_green
            if(USE_OTB):
                if(OTB_DETECT_PEOPLE_ONLY):
                    detectionsEveryFrame = FindSpecific(detectionsEveryFrame, 'person')     
        
            if(drawYOLO and not PLOT_AND_COMPARE_CENTERS):
                image = cvDrawBoxes(detectionsEveryFrame, image,COLOR)
            
            if(PLOT_AND_COMPARE_CENTERS):
                DetectionsEveryFrameBuffer[(frameIndex)%YOLO_DET_SKIP] = detectionsEveryFrame.copy()
                #for el in detections:
                AllDetections.append(detectionsEveryFrame.copy())
                image = cvDrawCenters(AllDetections, image,COLOR,5)
        
        
        #If it is time to redetect with darknet
        if(frame%YOLO_DET_SKIP==0 or detections is None or frame ==1):
        
            #PErform darknet detection
            #IT WAS: thresh=0.25
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=YOLO_DET_THRESH, hier_thresh=HI_THRESH, nms=NMS)
           # 
            if(USE_OTB):
                if(OTB_DETECT_PEOPLE_ONLY):
                    
                    detections = FindSpecific(detections, 'person')
                 
        COLOR = COLOR_green
        
        if(drawYOLO and not PLOT_AND_COMPARE_CENTERS):
            image = cvDrawBoxes(detections, image,COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
###################################################################################################################
    ##MV box copy/adjust
        ##Based on detection skip param
    
        SanityBuffer[(frame-1)%YOLO_DET_SKIP] = frameIndex
        
        #If boxes were redetected, copy them to mvboxes
        if(frame%YOLO_DET_SKIP==0 or MVBoxes is None):
            
            MVBoxes =  AssocDetections(detections)
            if(not USE_MB_MOTION):
                Mv_info = (good_new, good_old, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ)
                #
            else:
                ##MACROBLOCK MOTION WORK
                xMotionVector, yMotionVector, MB_LISTS = GetMacroBlockMotionVectors(MB_PARAM, MB_LISTS, frame_gray, frame_gray_prev)
            
                Mv_info = (xMotionVector, yMotionVector, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ)
            
            
            MVBuffer[(frame-1)%YOLO_DET_SKIP] = deepcopy(Mv_info)
            
        else:
            MVBoxes, dbgFrame,addedToFrame = UpdateMvBoxes(MVBoxes, good_new, good_old, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ, DEBUG_OBJECTS, image, mask)
        COLOR = COLOR_red
        
        
        #Draw the mv boxes on screen in red
        if(Draw_MV_BOXES and not PLOT_AND_COMPARE_CENTERS):
            if(dbgFrame is None):
                image = cvDrawBoxes(MVBoxes, image,COLOR)
            else:
                image = cvDrawBoxes(MVBoxes, dbgFrame,COLOR)
        
        if(PLOT_AND_COMPARE_CENTERS):
            #for el in MVBoxes:
            AllMVBoxes.append(MVBoxes.copy())
            image = cvDrawCenters(AllMVBoxes, image,COLOR)
        
###################################################################################################################

        #OTB SINGLE LINE GT LOAD
        if(USE_OTB):
            #Load in OTB data
            readIn = otb_file.readline()
            readIn = readIn.replace(',',' ')
            readIn = readIn.split()
            
            #Get params
            x_gt = int(readIn[0])
            y_gt = int(readIn[1])
            w_gt = int(readIn[2])
            h_gt = int(readIn[3])
            
            #scale boxes based on new limits on frame
            x_gt = int( round(otb_x_scale*x_gt))
            w_gt = int( round(otb_x_scale*w_gt))
            y_gt = int( round(otb_y_scale*y_gt))
            h_gt = int( round(otb_y_scale*h_gt))
            
            #Convert to normal representation. OTB serves x, y, w, h where xy is top left. We use the center
            x_gt = int(round(x_gt + w_gt/2))
            y_gt = int(round(y_gt + h_gt/2))
            
            #Make the detection item
            gt_box = ("", 0, (x_gt, y_gt, w_gt, h_gt))
            otblist.append(gt_box)

#           MVBOX EVAL: This matches a ground truth box with the closest overlapping detection and returns a pair            
            matches= MatchMVboxToYoloNew(MVBoxes, otblist, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y)
            
#           YOLO EVAL
            #matches= MatchMVboxToYoloNew(detections,otblist, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y)
            #
            
            ## Now, with match to ground truth, do stuff
            
            if(not PLOT_AND_COMPARE_CENTERS):
                pass
                #This will just show you the matches visually
#                image = DrawMatchesDiffColors(matches, detections, image)
            
            #Default vals
            center_dist = -1
            iou_value = -1.01
        
            #FOR frame:
            
            #Needs to be done
            #Calulate IOU
                #= IOU_CALC(matches)
                
            #Calculate centroid distance
            distanceList, garbage = CalcDistances(matches)
            
            #Print the distance if it was recorded
            if(len(distanceList)>0):
                center_dist = distanceList[0]
                print(center_dist)
            
            #Write results to file
            success_fptr.write( '%d %f \n' % (frame, iou_value))
            prec_fptr.write(    '%d %f \n' % (frame, center_dist))
        
        #If plot and compare centers, this is where the boxes are matched and distances are calulated
        if(PLOT_AND_COMPARE_CENTERS and not DETECT_DELAY):
        
            #Add this frames matched points to list
            AllMatchedBoxes_MV_YOLO.append(MatchMVboxToYoloNew(AllMVBoxes[frame-1], AllDetections[frame-1], MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y))
        
            #Match detections every frame to otb gt
            if(USE_OTB):
                AllMatchedBoxes_YOLO_to_GT.append(MatchMVboxToYoloNew(otblist, AllDetections[frame-1], MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y))
                
                AllMatchedBoxes_MV_to_GT.append(MatchMVboxToYoloNew(otblist, AllMVBoxes[frame-1], MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y))
            
            FrameDistancesMVYOLO, garbage = CalcDistances(AllMatchedBoxes_MV_YOLO[frame-1])
            
            #Provide list of calulated distances
            if(USE_OTB):
                FrameDistancesMV, garbage = CalcDistances(AllMatchedBoxes_MV_to_GT[frame-1])
                FrameDistancesYOLO, garbage = CalcDistances(AllMatchedBoxes_YOLO_to_GT[frame-1])
            else:
                FrameDistancesMV=[]
                FrameDistancesYOLO=[]
            
            numDetectionsMV = len(AllMVBoxes[frame-1])
            numDetectionsYOLO = len(AllDetections[frame-1])
            
            #If there were frames without detections, give it a -10
            if(not numDetectionsMV==0):
                AverageDistMVYOLO = sum(FrameDistancesMVYOLO)/numDetectionsMV
                AverageDistMV = sum(FrameDistancesMV)/numDetectionsMV
            else:
                print("PLOT and compare centers logged number of detections as 0")
            
                AverageDistMVYOLO = -10
                AverageDistMV = -10
                
            if(not numDetectionsYOLO==0):
                AverageDistYOLO = sum(FrameDistancesYOLO)/numDetectionsYOLO                
            else:
                print("PLOT and compare centers logged number of detections as 0")

                AverageDistYOLO = -10

            #Add this frames distances to distance list
            FrameCumulativeDriftMVYOLO.append(AverageDistMVYOLO)
            FrameCumulativeDriftMV.append(AverageDistMV)
            FrameCumulativeDriftYOLO.append(AverageDistYOLO)
            

        if(SHOW_OTB_GT):
            COLOR = (255,32,64)
            
            image = cvDrawBoxes(otblist, image,COLOR)
            
        ##########################################
        otblist.clear()

        if(PRINT_FRAMERATE):
        
            frameRateCur = (1/(time.time()-prev_time))
            sys.stdout.write("Frame Rate: %f     \r" % frameRateCur)
            sys.stdout.flush()
            frameRateArr.append(frameRateCur)
        
        ################ BIG VIEW ##
        #Magnify it eyes
        #fin_Im = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
        #cv2.imshow('Frame', fin_Im)
        ################          ##
        
        cv2.line(image, (int(image.shape[1]/2), 0),(int(image.shape[1]/2), image.shape[0]), COLOR_blue, 1, 1)
        cv2.line(image, (0, int(image.shape[0]/2)),(image.shape[1], int(image.shape[0]/2) ), COLOR_blue, 1, 1)
        
        if(not DETECT_DELAY):
        
            cv2.imshow('Frame', image)
            #cv2.waitKey(3)
            if(cv2.waitKey(3) & 0xFF == ord('q')):
                break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    #plot center drift values
    if (PLOT_AND_COMPARE_CENTERS and not DETECT_DELAY):
        del loopsArr[-1]
        
        if(USE_OTB):
            NewFrameCumulativeDriftYOLO = np.array(FrameCumulativeDriftYOLO)
            NewFrameCumulativeDriftYOLO = NewFrameCumulativeDriftYOLO[NewFrameCumulativeDriftYOLO !=-10]
            AvgTotalDistYOLO = np.sum(NewFrameCumulativeDriftYOLO)/len(loopsArr)
            plt.plot([0, loopsArr[-1]], [AvgTotalDistYOLO, AvgTotalDistYOLO], 'k-', color = 'g', linewidth=1)
            plt.plot(loopsArr, FrameCumulativeDriftYOLO, 'k-', color = 'g', linewidth=1)
            
            NewFrameCumulativeDriftMV = np.array(FrameCumulativeDriftMV)
            NewFrameCumulativeDriftMV = NewFrameCumulativeDriftMV[NewFrameCumulativeDriftMV !=-10]
            AvgTotalDistMV = np.sum(NewFrameCumulativeDriftMV)/len(loopsArr)
            plt.plot([0, loopsArr[-1]], [AvgTotalDistMV, AvgTotalDistMV], 'k-', color = 'r', linewidth=1)
            plt.plot(loopsArr, FrameCumulativeDriftMV, 'k-', color = 'r', linewidth=1)

        plt.xlabel('Current Frame')
        plt.ylabel('Pixel Distance')
        plt.title("Average Per-Box Centroid Drift MVBox to YOLO Box Chart NO Delay")
        
        A1ResultsPath = "./A1Results/"
        nameToSave= A1ResultsPath+ "AvCenterDrift" + "_yolo_of_" + str(YOLO_DET_SKIP) + path.split("\\")[-1] + ".png"
        plt.savefig(nameToSave)
    
    #plot framerate values
    if (PRINT_FRAMERATE):
        
        if(len(loopsArr) != len(frameRateArr)):
            print("Adjust array lengths")
            del loopsArr[-1]
        plt.plot(loopsArr, frameRateArr, label='Framerate Values')
        plt.legend()
        plt.show()
    
    if(USE_OTB):
        success_fptr.close()
    
        prec_fptr.close()

    #release memory
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FixVSPath()
    color = np.random.randint(0,255,(DetectionPoints,3))
    import dark_motion as darknet
    YOLO()
