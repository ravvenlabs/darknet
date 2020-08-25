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



#openCV static info
path = "./data/test.mp4"
#path = "./data/simpler_trim.mp4"
#path = "./data/two_min_alps_traffic.mp4"

####################################################################

#USE OTB DATA

#Using OTB will cuase the program to read in OTB data which is a CV benchmark set
USE_OTB = True

        
if(USE_OTB):
    path = ".\data\OTB_data\stationary\Walking2\otb_Walking2.avi"
    otb_gt_file = ".\data\OTB_data\stationary\Walking2\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\stationary\Walking\otb_Walking.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Walking\groundtruth_rect.txt"
    
    
    #path = ".\data\OTB_data\stationary\Crossing\otb_crossing.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Crossing\groundtruth_rect.txt"
    
    #path = ".\data\OTB_data\stationary\Subway\otb_Subway.avi"
    #otb_gt_file = ".\data\OTB_data\stationary\Subway\groundtruth_rect.txt"
    
    

    OTB_DETECT_PEOPLE_ONLY = True

    #WHEN USING! make sure the path is to an otb data folder as well

    #plot otb ground truth
    SHOW_OTB_GT = False
    
    
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
    
    
#    pdb.set_trace()

else:
    SHOW_OTB_GT = False



#Frames till next optical flow point refresh
OF_DET_SKIP = 4

#Force darknet to predict and redraw every frame
ALWAYS_REDRAW_Redetect = False

#frames till next yolo call 
YOLO_DET_SKIP = 10

#PRint and plot framerate
PRINT_FRAMERATE = False

#Rectangles for box movement and crosshairs on MV in said rectangle (And print locations of each)
DEBUG_OBJECTS = False

#MV inside of rectangle buffer for outside of rectangle [units of pixels]
MV_RECT_BUFFER_VERT = 10
MV_RECT_BUFFER_HORZ = 10

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
CV_LINES_ON = False

#more objects detected and tracked with O.F.
DetectionPoints = 250

#draw yolo bboxes (every frame green)
drawYOLO = True

#Draw motion-vector-propelled boxes
Draw_MV_BOXES = True

#Calulate the Centroid drift
CALC_MV_YOLO_CENTER_DRIFT = False

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
#configPath = "./cfg/yolov3.cfg"
#weightPath = "./weights/yolov3.weights"
#metaPath = "./cfg/coco.data"

#yolov2 tiny
configPath = "./cfg/yolov2-tiny.cfg"
weightPath = "./weights/yolov2-tiny.weights"
metaPath = "./cfg/coco.data"

#yolov3 tiny
#configPath = "./cfg/yolov3-tiny.cfg"
#weightPath = "./weights/yolov3-tiny.weights"
#metaPath = "./cfg/coco.data"


#-->VOC is horrible currently
#configPath = "./cfg/yolov2-tiny-voc.cfg"
#weightPath = "./weights/yolov2-tiny-voc.weights"
#metaPath = "./cfg/voc.data"

###########


#####################################################################


frame = 0
# ####################

#This function just detects where the file is being run from and adjusts paths to hit required folders 
def FixVSPath():

    subFolder = (os.getcwd()).split("\\")[-1]
    print(os.getcwd())
    if(not subFolder == "x64"):

        print("VS detected subdir change needed")
        print(os.getcwd())
        newTarg = os.path.join(os.getcwd(), "build" , "darknet", "x64")
        os.chdir(newTarg)
        print(os.getcwd())
        subFolder = (os.getcwd()).split("\\")[-1]
        print("Done")

    else:
        print("No subdir change needed")


#Exsiting function from darknet
#This just converts a rectangles center (x,y), width and height to the min and max x and y values 
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
    
#otb data format is x,y,w,h where (x,y) is the top left
def convertBackOTB(x, y, w, h):
    xmin = int(round(x))
    xmax = int(round(x + (w)))
    ymin = int(round(y))
    ymax = int(round(y + (h)))
    return xmin, ymin, xmax, ymax


#Exsiting function from darknet
#This function draws boxes on the frame where detections are
def cvDrawBoxes(detections, img, COLOR):
    red,green,blue = COLOR
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (red,green,blue), 1)
        #cv2.putText(img,
        #            detection[0].decode() +
        #            " [" + str(round(detection[1] * 100, 2)) + "]",
        #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            [red,green,blue], 2)
    return img
    
#Exsiting function from darknet
#This function draws boxes on the frame where detections are
def cvDrawBoxesOTB(detections, img, COLOR):
    red,green,blue = COLOR
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBackOTB(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (red,green,blue), 1)
    #    cv2.putText(img,
    #                detection[0].decode() +
    #                " [" + str(round(detection[1] * 100, 2)) + "]",
    #                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                [red,green,blue], 2)
    return img


netMain = None
metaMain = None
altNames = None

#This will just draw one box in a list. It is for debugging without too much clutter
def cvDrawOneBox(detection, img, COLOR, otb_draw=False):

    red,green,blue = COLOR

    #for detection in detections:
    x, y, w, h = detection[2][0],\
        detection[2][1],\
        detection[2][2],\
        detection[2][3]
    if(otb_draw):        
        xmin, ymin, xmax, ymax = convertBackOTB(float(x), float(y), float(w), float(h))
    else:
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(img, pt1, pt2, (red, green, blue), 2)
    #cv2.putText(img,
    #            detection[0].decode() +
    #            " [" + str(round(detection[1] * 100, 2)) + "]",
    #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #            [0, 255, 0], 2)
#    break
    return img


netMain = None
metaMain = None
altNames = None
#Copy detections
def AssocDetections(detections):
    
    MV_and_Detections = []
    
    #id = 0
    for detection in detections:
        MV_and_Detections.append( detection) #(id, detection, list()) ) 
        #id+=1
        
    return MV_and_Detections
    
#Determine whether a point is inside a rectangle (Error buffer is taken into account too)
def InsideRect(a,b,x,y,w,h, bufferV,bufferH):
    return (a>=(x-bufferH) and a<=(x+w+bufferH) and b>=(y-bufferV) and b<=(y+h+bufferV))
    
#Move the motion vector boxes 
#takes in the new detections and the new and old optical flow points
def UpdateMvBoxes(detections, newFramePoints, oldFramePoints, dbgFrame=None, mask = None):
    
    element = 0
    addedToFrame = False
    
    #For each detection
    for detection in detections:
        
        #(id, detection, mvs) = packed
        #pdb.set_trace()
        #print(id)
        
        DistX = 0
        DistY = 0
        
        #unpack this detection
        name, detsProb, (x,y,w,h) = detection
           
        total = 0
        
        #for each set of points in new and old
        for i,(new,old) in enumerate(zip(newFramePoints, oldFramePoints)):
                
                a,b = new.ravel()
                c,d = old.ravel()
               
                #pdb.set_trace()
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                #frame_read = cv2.circle(frame_read,(a,b),5,color[i].tolist(),-1)
        
                #if new point is inside rectangle
                if(InsideRect(a,b,x,y,w,h, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ)):
                
                    total+=1
                    #mask = cv2.line(mask, (int(0),int(y)),(int(1000),int(y+h)), (0,255,150), 5)
                    #mask = cv2.line(mask, (int(x),int(0)),(int(x+w),int(1000)), (0,255,150), 5)
                    #pdb.set_trace()
                    
                    #For single frame debug stop
                    addedToFrame = True
                    
                    if(DEBUG_OBJECTS):
                        xmin, ymin, xmax, ymax = convertBack( float(x), float(y), float(w), float(h))
                        pt1 = (xmin, ymin)
                        pt2 = (xmax, ymax)
                        mask = cv2.rectangle(mask, (pt1),(pt2), (255,0,0), 5)
                        mask=cv2.line(mask, (int(a), 0),(int(a), mask.shape[0]), (255, 255, 0), 1, 1)
                        mask=cv2.line(mask, (0, int(b)),(mask.shape[1], int(b) ), (255, 255, 0), 1, 1)
#                       #pdb.set_trace()
                        print("Rectangle at " ,pt1 ," by " , pt2)
                        print("OpenCV point is at " ,a ," by " , b)
                    else:
                        pass
                
                    #Update the x and y change based on new and old point movement
                    DistX = DistX +  a-c
                    DistY = DistY + b-d
    
        #Divide distance by number of mvs in box
        if(total>0):
            DistX = DistX / total
            DistY = DistY / total
                
        #If the x or y needs to be adjusted because shift is nonzero, call move function
        if(DistX != 0 or DistY != 0):
            #pdb.set_trace()
            
            #This function actually accesses the detections properties
            NewDetection = MoveDetection(detection, DistX, DistY)
            #replace
            detections[element] = NewDetection
        element+=1
        
    return detections, dbgFrame, addedToFrame

#This function actually accesses the detections struct paramss and updates them
def MoveDetection(detection, dx,dy):
    #x -> detection[2][0]
    #y -> detection[2][1]
    name, detsProb, (x,y,w,h) = detection
    #x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
    #pdb.set_trace()
    x+=dx
    y+=dy
    #unpack 
    #res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    #detection[2][0] = x 
    #detection[2][1] = y 
    #detection[2][2] = w 
    #detection[2][3] = h
    detection = (name, detsProb, (x,y,w,h))
    
    return detection

#This function will return true if one rectangle overlaps with another
#pass in top left and bottom right tuples of points
#top left 1, bot right 1
#top left 2, bot right 2
#rect 1 is mvbox
#rect 2 is yolo box
def rectOverlap(rect1,rect2):
    
    (r1_p1,r1_p2) = rect1
    
    (r2_p1,r2_p2) = rect2
    
    #[0] is x, [1] is y
    
    #if overlap left or right
    if(r1_p1[0] - MV_YOLO_ASSOCIATION_BUFFER_X >= r2_p2[0] or r2_p1[0] - MV_YOLO_ASSOCIATION_BUFFER_X >= r1_p2[0]):
        #print("left or right fail")
        #print("Failed LR")
        return False
    
    #if overlap top or bottom
    if(r1_p1[1] - MV_YOLO_ASSOCIATION_BUFFER_Y >= r2_p2[1] or r2_p1[1]- MV_YOLO_ASSOCIATION_BUFFER_Y >= r1_p2[1]):
        #print("Top bottom fail")
        #print("Top bottom fail")
        #print("Failed TB")
        return False
    
    return True
    
#This function is for matching a new detection to an existing box for an object
#It will go thru all existing boxes and then see if there are any new detection boxes that overlap the old mv box
#If there are, candidates distances are calculated and the closest box is matched to the existing box
def MatchMVboxToYoloNew(detections, mvBoxes, dbgimg=None):
    pass
    mvBoxesWithYoloID = []
    candidate = []
    dist = 10000
    
    #for motion vector boxes
    for mvbox in mvBoxes:
    
        name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mvbox
        xmin_mv, ymin_mv, xmax_mv, ymax_mv = convertBack(
                    float(x_mv), float(y_mv), float(w_mv), float(h_mv))
        
        #get corner points and put in tuple
        pt1_mv = (xmin_mv, ymin_mv)
        pt2_mv = (xmax_mv, ymax_mv)
        count = 0
        at_least_one_match = False
    
        #For all new detections
        for yolo_det in detections: 
            #unpack
            name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo_det
            #pdb.set_trace()
            xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = convertBack(
                    float(x_yolo), float(y_yolo), float(w_yolo), float(h_yolo))
            
            #get corner points and put in tuple
            pt1_yolo = (xmin_yolo, ymin_yolo)
            pt2_yolo = (xmax_yolo, ymax_yolo)

            count +=1

            #If there is an overlap of an mv and a yolo box
            if( rectOverlap( (pt1_mv, pt2_mv), ( pt1_yolo, pt2_yolo) ) ):
                #sys.stdout.write("Rect overlap\r")
                #sys.stdout.flush()
                at_least_one_match = True
                #overlap found, make match
                #mvBoxesWithYoloID.append((mvbox, yolo_det))
                #break
                #calulate distance fo the boxes
                dist = DiagDist(x_mv,y_mv, x_yolo,y_yolo)              
                #add it to the list
                candidate.append((yolo_det, dist))
  
            else:
                #sys.stdout.write("              \r" )
                #sys.stdout.flush()
                pass
        
        #find closest box to pair with
        if(at_least_one_match):
            least = 0
            least_val = 10000
            min_ct = 0
            
            #for each candidate box determine which is the closest
            for yolo_det_dist in candidate:
                color = (0,0,255)
                #cvDrawOneBox(yolo_det, dbgimg, color)
                yolo_det,dist = yolo_det_dist
                
                if(dist<least_val):
                    least = min_ct
                    least_val = dist
        #            
                min_ct+=1
            
        #    pdb.set_trace()
            
            #Add matched pair to return list
            mvBoxesWithYoloID.append((candidate[least][0], mvbox))
            candidate.clear()

    if(dbgimg is None):
        return mvBoxesWithYoloID
    else:
        #print("Returning with dbg image")
        
        return mvBoxesWithYoloID, dbgimg

#This is a distance calculation between two points
def DiagDist(x1,y1,x2,y2):
    #return center point
    
    dx = x1-x2
    dy= y1-y2
    dist = math.sqrt( dx*dx+dy*dy )
    
    return dist
    
#This is for the drift calulations. It calulates the distances between each pair of boxes
def CalcDistances(matches):
    
    
    distList = []
    totalDriftAmount = 0
    
    for match in matches:
        #pdb.set_trace()
        #Center = centroid(p1)
        #center2 = centroid(p2)
        
        mvBox, yoloBox = match
        
        name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mvBox
        
        name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yoloBox
    
        dist = DiagDist(x_mv,y_mv, x_yolo,y_yolo)
        
        distList.append(dist)
    
        #Get yolobox diag
    
        xmin, ymin, xmax, ymax = convertBack(x_yolo,y_yolo,w_yolo,h_yolo)
        totalDriftAmount += (dist)/(DiagDist( xmin,ymin,xmax,ymax))
    
    
    
    return distList, totalDriftAmount


#This function draws a white line between 2 boxes
def cvDrawLinkCenter(mv,yolo, image):
    
    name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mv
        
    name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo
    
    image = cv2.line(image, (int(x_mv),int(y_mv)),(int(x_yolo),int(y_yolo)), (255,255,255), 2)
    
    
    return image

#This function will go through a list of pairs and draw the connecting line.
#It will also show the paired boxes in different colors for easy inspection of pairing
def DrawMatchesDiffColors(matches, detections, image, otb_draw=False):
    #cnt=0
    
    for match in matches:
        box1, box2 = match
        #cnt+=1
        #if(cnt>1):
        #    pdb.set_trace()
        #COLOR = (0,255,255)
        
        ColorList = list(np.random.choice(range(106), size=3)+100)

        
        COLOR = (ColorList[0].item(), ColorList[1].item(), 0)
        #COLOR = (0,0,255)
        #COLOR2 = (255,0,0)
        
        #COLOR2 = (COLOR[0]+50, COLOR[1]+50, COLOR[2]+50)
#       
        image = cvDrawLinkCenter(box1,box2, image)

 #       pdb.set_trace()
        
        #if useotb is true, box one is otb
        if(otb_draw):
            image = cvDrawOneBox(box1, image, COLOR, otb_draw=True)
            #image = cvDrawBoxesOTB([box1], image,COLOR)
        else:
            image = cvDrawOneBox(box1, image, COLOR)
            
        image = cvDrawOneBox(box2, image, COLOR)
        
    return image

#Main loop
def YOLO():

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
    #cap = cv2.VideoCapture(0)
#    cap = cv2.VideoCapture("test.mp4")
    cap = cv2.VideoCapture(path)
    
    #Record factors to scale by
    if(USE_OTB):
        otb_width  = cap.get(3) 
        otb_height = cap.get(4) 
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
    
    
    #FRAME READ MAIN LOOP
    #
    #####################
    
    ###
    #OpenCV get first frame
    #
    
    #get frame
    ret, frame_read = cap.read()
    if(USE_OTB):
        otb_file = open(otb_gt_file)
        otb_file.readline()
        
    
    #resize to darknet preference
    frame_read = cv2.resize(frame_read, (darknet.network_width(netMain),
                                    darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
    #Get the first frame in gray
    old_gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
    
    #Use opencv to gather the points to track with lk
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame_read)
    
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
    breakAt = 1000
    
    
    #While the video is open
    while cap.isOpened():
        addedToFrame = False
        frame=frame+1
        loopsArr.append(frame)
        if(SLOW_MODE):
            time.sleep(0.05)
        
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
        
        
        prev_time = time.time()
        ret, frame_read = cap.read()
        
        #Video failed to return another frame
        if(not ret):
            break
        
        #resize the frame for darknet
        frame_read = cv2.resize(frame_read, (darknet.network_width(netMain),
                                    darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        
        
        ###############################MVs#########################
            ##optical flow code#
        
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
        
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
          
        #If it is time to redetect with darknet
        if(frame%YOLO_DET_SKIP==0 or detections is None or ALWAYS_REDRAW_Redetect):
        
            #PErform darknet detection
            #IT WAS: thresh=0.25
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=YOLO_DET_THRESH, hier_thresh=HI_THRESH, nms=NMS)
           # pdb.set_trace()
            if(USE_OTB):
                if(OTB_DETECT_PEOPLE_ONLY):
                    
                    pop_tracker = 0
                    
                    det_len = len(detections)
                    
                    for detection in range(0, det_len):
                        #if it isa not a person
    #                    print(detections[0][0].decode())
                        if(not detections[pop_tracker][0].decode() == 'person'):
                            
                            detections.pop(pop_tracker)
                            pop_tracker-=1
                        pop_tracker+=1
                        
                    #pdb.set_trace()
                    
                    #print(detections)
       
        COLOR = (0,255,0)
        
        if(drawYOLO):
            image = cvDrawBoxes(detections, image,COLOR)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
###################################################################################################################
    ##MV box copy/adjust
        ##Based on detection skip param
    

        #If boxes were redetected, copy them to mvboxes
        if(frame%YOLO_DET_SKIP==0 or MVBoxes is None):
            MVBoxes =  AssocDetections(detections)
            #sys.stdout.write("MVBoxes assigned\n")
        #else, update them with motion vectors
        else:
            MVBoxes, dbgFrame,addedToFrame = UpdateMvBoxes(MVBoxes, good_new, good_old, image, mask)
        COLOR = (0,0,255)
        #pdb.set_trace()
        
        #Draw the mv boxes on screen in red
        if(Draw_MV_BOXES):
            if(dbgFrame is None):
                image = cvDrawBoxes(MVBoxes, image,COLOR)
            
            else:
                image = cvDrawBoxes(MVBoxes, dbgFrame,COLOR)
        else:
            pass
###################################################################################################################
        
        
        ##########################################
        ## OTB STUFF ##
        
        
        #OTB SINGLE LINE GT LOAD
        if(USE_OTB):
            #Load in OTB data
            readIn = otb_file.readline()
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
            matches= MatchMVboxToYoloNew(MVBoxes, otblist)
            
#           YOLO EVAL
            #matches= MatchMVboxToYoloNew(detections,otblist)
            #pdb.set_trace()
            
            ## Now, with match to ground truth, do stuff
            
            #This will just show you the matches visually
            image = DrawMatchesDiffColors(matches, detections, image, otb_draw=False)
        
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
            #pdb.set_trace()
        
        if(SHOW_OTB_GT):
            COLOR = (255,32,64)
            image = cvDrawBoxesOTB(otblist, image,COLOR)
            
        ##########################################
        otblist.clear()
        
    #Calculate center drift between detections and MVboxes
        #pdb.set_trace()
        if (CALC_MV_YOLO_CENTER_DRIFT):
            matches, image = MatchMVboxToYoloNew(detections,MVBoxes, image)
            image = DrawMatchesDiffColors(matches, detections, image)
        
            #CALC_DISTANCES()
            distances, totalError = CalcDistances(matches)
            cumulitiveDrift +=totalError
        
        
        ##########################################
        
        
        #print data to cmd line
        if(CALC_MV_YOLO_CENTER_DRIFT):
            sys.stdout.write("Drift amount: %f     \r" % totalError )
            sys.stdout.flush()            
            driftArr.append(totalError)
            
        if(PRINT_FRAMERATE):
        
            frameRateCur = (1/(time.time()-prev_time))
            sys.stdout.write("Frame Rate: %f     \r" % frameRateCur)
            sys.stdout.flush()
            frameRateArr.append(frameRateCur)
        
        ################ BIG VIEW ##
        #MAgnify it eyes
        #fin_Im = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
        #cv2.imshow('Frame', fin_Im)
        ################          ##
        
        cv2.line(image, (int(image.shape[1]/2), 0),(int(image.shape[1]/2), image.shape[0]), (255, 0, 0), 1, 1)
        cv2.line(image, (0, int(image.shape[0]/2)),(image.shape[1], int(image.shape[0]/2) ), (255, 0, 0), 1, 1)
        
        #cv2.line(image, (0, int(400)),(image.shape[0], int(400) ), (255, 0, 0), 1, 1)
        #cv2.line(image, (0, int(200)),(image.shape[0], int(200) ), (0, 255, 0), 1, 1)
        #cv2.line(image, (0, int(100)),(image.shape[0], int(100) ), (0, 0, 255), 1, 1)
        
        #print("Image is " ,image.shape[0] ," by " , image.shape[1])
        
        if(addedToFrame):   
            #pdb.set_trace()
            pass
        cv2.imshow('Frame', image)
        
        
        
##################################################################################################################################
        
        #Do map testing down here, possibly
        
        
##################################################################################################################################        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        #cv2.waitKey(3)
        if(cv2.waitKey(3) & 0xFF == ord('q')):
            break
            
        if(frame > breakAt and ( CALC_MV_YOLO_CENTER_DRIFT or PRINT_FRAMERATE)):
            break
    
    #plot center drift values
    if (CALC_MV_YOLO_CENTER_DRIFT):
        plt.plot(loopsArr, driftArr, label='Drift Values')
    #    plt.legend()
        plt.show()
    
    #plot framerate values
    if (PRINT_FRAMERATE):
        plt.plot(loopsArr, frameRateArr, label='Framerate Values')
        plt.legend()
        plt.show()
    
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
