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

#Frames till next optical flow point refresh
OF_DET_SKIP = 4

ALWAYS_REDRAW = False



#frames till next yolo call 
YOLO_DET_SKIP = 1

PRINT_FRAMERATE = True

PRINT_DRIFT_CALC = False

#Rectangles for box movement and crosshairs on MV in said rectangle (And print locations of each)
DEBUG_OBJECTS = False

#MV inside of rectangle buffer for outside of rectangle [units of pixels]
MV_RECT_BUFFER_VERT = 10
MV_RECT_BUFFER_HORZ = 10

#Yolo accuracy required to make a bbox
YOLO_DET_THRESH = .35
HI_THRESH=.5
NMS=.30
#Insert a delay from one frame to the next
SLOW_MODE =False
#with VOC
#YOLO_DET_THRESH = .10

MV_YOLO_ASSOCIATION_BUFFER_X = 5
MV_YOLO_ASSOCIATION_BUFFER_Y = 5


#put circlews on most recent O.F. point
CV_CIRCLE_ON = False
#put lines on for O.F.
CV_LINES_ON = False

#more objects detected and tracked
DetectionPoints = 250

#draw yolo bboxes (every frame green)
drawYOLO = True

Draw_MV_BOXES = True

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



    



#####################################################################


loops = 0
# ####################


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


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxesFromMVList(detections, img, COLOR):
    red,green,blue = COLOR
    for detection in detections:
        
        #throw, detection, out = detection 
    
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
    #    cv2.putText(img,
    #                detection[0].decode() +
    #                " [" + str(round(detection[1] * 100, 2)) + "]",
    #                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                [red,green,blue], 2)
    return img


netMain = None
metaMain = None
altNames = None

def cvDrawOneBox(detection, img, COLOR):

    red,green,blue = COLOR

    #for detection in detections:
    x, y, w, h = detection[2][0],\
        detection[2][1],\
        detection[2][2],\
        detection[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
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

def AssocDetections(detections):
    
    MV_and_Detections = []
    
    #id = 0
    for detection in detections:
        MV_and_Detections.append( detection) #(id, detection, list()) ) 
        #id+=1
        
    return MV_and_Detections

def InsideRect(a,b,x,y,w,h, bufferV,bufferH):
    
    return (a>=(x-bufferH) and a<=(x+w+bufferH) and b>=(y-bufferV) and b<=(y+h+bufferV))
    
    
def UpdateMvBoxes(detections, newFramePoints, oldFramePoints, dbgFrame=None, mask = None):
    
    element = 0
    addedToFrame = False
    for detection in detections:
        
        #(id, detection, mvs) = packed
        #pdb.set_trace()
        #print(id)
        
        DistX = 0
        DistY = 0
        
        #unpack this detection
        name, detsProb, (x,y,w,h) = detection
        
        #added = False
        
        total = 0
        
        for i,(new,old) in enumerate(zip(newFramePoints, oldFramePoints)):
                a,b = new.ravel()
                c,d = old.ravel()
                
                
                
                #pdb.set_trace()
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                #frame_read = cv2.circle(frame_read,(a,b),5,color[i].tolist(),-1)
        
                #if old point inside rectangle
                
                if(InsideRect(a,b,x,y,w,h, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ)):
                
                    total+=1
                
                    #mask = cv2.line(mask, (int(0),int(y)),(int(1000),int(y+h)), (0,255,150), 5)
                    #mask = cv2.line(mask, (int(x),int(0)),(int(x+w),int(1000)), (0,255,150), 5)
                    #pdb.set_trace()
                    xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    
                    addedToFrame = True
                    
                    if(DEBUG_OBJECTS):
                        mask = cv2.rectangle(mask, (pt1),(pt2), (255,0,0), 5)
                        mask=cv2.line(mask, (int(a), 0),(int(a), mask.shape[0]), (255, 255, 0), 1, 1)
                        mask=cv2.line(mask, (0, int(b)),(mask.shape[1], int(b) ), (255, 255, 0), 1, 1)
#                    
                    #pdb.set_trace()
                        print("Rectangle at " ,pt1 ," by " , pt2)
                    
                        print("OpenCV point is at " ,a ," by " , b)
                    
                    else:
                        pass
                    
                    
                    
                    #added = True
                    #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    #dbgFrame = cv2.circle(dbgFrame,(a,b),5,color[i].tolist(),-1)
                    #pdb.set_trace()
                    DistX = DistX +  a-c
                    DistY = DistY + b-d
    
    
        #if(added):
        #    if(dbgFrame is not None):
        #        pdb.set_trace()
                #cv2.imshow('Frame', dbgFrame)
    
        if(total>0):
    
            DistX = DistX / total
            DistY = DistY / total
                
        if(DistX != 0 or DistY != 0):
            #pdb.set_trace()
            NewDetection = MoveDetection(detection, DistX, DistY)
        
            #repacked = (id, NewDetection, mvs)
        
            detections[element] = NewDetection
        
        element+=1
        
        
        
    return detections, dbgFrame, addedToFrame

def MoveDetection(detection, dx,dy):
    #x -> detection[2][0]
    #y -> detection[2][1]
    name, detsProb, (x,y,w,h) = detection
    x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
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
    

def MatchMVboxToYoloNew(detections, mvBoxes, dbgimg=None):
    pass
    mvBoxesWithYoloID = []
    candidate = []
    dist = 10000
    
    for mvbox in mvBoxes:
        name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mvbox
        
        xmin_mv, ymin_mv, xmax_mv, ymax_mv = convertBack(
                    float(x_mv), float(y_mv), float(w_mv), float(h_mv))
        pt1_mv = (xmin_mv, ymin_mv)
        pt2_mv = (xmax_mv, ymax_mv)
        count = 0
        at_least_one_match = False
        
        for yolo_det in detections: 
            name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo_det
            #pdb.set_trace()
            xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = convertBack(
                    float(x_yolo), float(y_yolo), float(w_yolo), float(h_yolo))
            pt1_yolo = (xmin_yolo, ymin_yolo)
            pt2_yolo = (xmax_yolo, ymax_yolo)
        
            #print("Rectangles overlap")
            
            #print(count)
            count +=1
     
    #        color = (0,255,255)
     #       cvDrawOneBox(yolo_det, dbgimg, color)

            if( rectOverlap( (pt1_mv, pt2_mv), ( pt1_yolo, pt2_yolo) ) ):
                #sys.stdout.write("Rect overlap\r")
                #sys.stdout.flush()
                at_least_one_match = True
                #overlap found, make match
                #mvBoxesWithYoloID.append((mvbox, yolo_det))
                #break
                dist = DiagDist(x_mv,y_mv, x_yolo,y_yolo)
                
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
            
            mvBoxesWithYoloID.append((mvbox, candidate[least][0]))
            candidate.clear()
            
   #     color = (255,255,0)
   #     cvDrawOneBox(mvbox, dbgimg, color)
   
    #cv2.rectangle(dbgimg, (200,200),(250,250), (255,250,0), 3)
 
    if(dbgimg is None):
        return mvBoxesWithYoloID
    else:
        #print("Returning with dbg image")
        
        return mvBoxesWithYoloID, dbgimg

def DiagDist(x1,y1,x2,y2):
    #return center point
    
    dx = x1-x2
    dy= y1-y2
    dist = math.sqrt( dx*dx+dy*dy )
    
    return dist
    
    
    


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




def cvDrawLinkCenter(mv,yolo, image):
    
    name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mv
        
    name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo
    
    image = cv2.line(image, (int(x_mv),int(y_mv)),(int(x_yolo),int(y_yolo)), (255,255,255), 2)
    
    
    return image

def DrawMatchesDiffColors(matches, detections, image):
    for match in matches:
        mvbox, yolo_box = match
        
        #COLOR = (0,255,255)
        
        ColorList = list(np.random.choice(range(106), size=3)+100)

        
        COLOR = (ColorList[0].item(), ColorList[1].item(), 0)
        #COLOR = (0,0,255)
        #COLOR2 = (255,0,0)
        
        #COLOR2 = (COLOR[0]+50, COLOR[1]+50, COLOR[2]+50)
#       
        image = cvDrawLinkCenter(mvbox,yolo_box, image)

 #       pdb.set_trace()
        
 
        #image = cvDrawOneBox(mvbox, image, COLOR)
        #image = cvDrawOneBox(yolo_box, image, COLOR)
        
    return image

def YOLO():

    global metaMain, netMain, altNames
    #configPath = "./cfg/yolov4.cfg"
    #weightPath = "./weights/yolov4.weights"
    #metaPath = "./cfg/coco.data"
    
    #configPath = "./cfg/yolov3.cfg"
    #weightPath = "./weights/yolov3.weights"
    #metaPath = "./cfg/coco.data"

    configPath = "./cfg/yolov2-tiny.cfg"
    weightPath = "./weights/yolov2-tiny.weights"
    metaPath = "./cfg/coco.data"

    #configPath = "./cfg/yolov3-tiny.cfg"
    #weightPath = "./weights/yolov3-tiny.weights"
    #metaPath = "./cfg/coco.data"

    #-->VOC is horrible currently
    #configPath = "./cfg/yolov2-tiny-voc.cfg"
    #weightPath = "./weights/yolov2-tiny-voc.weights"
    #metaPath = "./cfg/voc.data"
    
    
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
    
    ret, frame_read = cap.read()
    frame_read = cv2.resize(frame_read, (darknet.network_width(netMain),
                                    darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
    old_gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame_read)
    
    loops=0
    
    frame_read= None
    MVBoxes = None
    dbgFrame = None
    detections = None
    maxx=0
    maxy=0
    
    cumulitiveDrift = 0
    
    driftArr = []
    frameRateArr = []
    loopsArr = []
    
    breakAt = 1000
    
    while cap.isOpened():
        addedToFrame = False
        loops=loops+1
        loopsArr.append(loops)
        if(SLOW_MODE):
            time.sleep(0.05)
        
        #every 10 start with new mask
        if(loops%OF_DET_SKIP==0):
            #Every 10 frames readjust
            # Take first frame and find corners in it
            ret, old_frame = ret, frame_read
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
            
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_read = cv2.resize(frame_read, (darknet.network_width(netMain),
                                    darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        
        
        
        ###############################MVs#########################
        
        frame_gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        NoneType=type(None)
        
        #make sure something is returned before we try and put it in the frame
        if(not isinstance(p1,NoneType)):
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                if(CV_LINES_ON):
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                if(CV_CIRCLE_ON):
                    frame_rgb = cv2.circle(frame_rgb,(a,b),5,color[i].tolist(),-1)

        image = cv2.add(frame_rgb,mask)
       
        #cv2.imshow('Demo', image)
        #################################################
        
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            
        if(loops%YOLO_DET_SKIP==0 or detections is None or ALWAYS_REDRAW):
        
        
            #IT WAS: thresh=0.25
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=YOLO_DET_THRESH, hier_thresh=HI_THRESH, nms=NMS)
            
        #image = cvDrawOneBox(detections, frame_resized)
        
        COLOR = (0,255,0)
        
        if(drawYOLO):
            image = cvDrawBoxes(detections, image,COLOR)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
###################################################################################################################
        
        #(id, detection, [])
        
        
        if(loops%YOLO_DET_SKIP==0 or MVBoxes is None):
            MVBoxes =  (detections)
            #sys.stdout.write("MVBoxes assigned\n")
        
        else:
        
            MVBoxes, dbgFrame,addedToFrame = UpdateMvBoxes(MVBoxes, good_new, good_old, image, mask)
        COLOR = (0,0,255)
        #pdb.set_trace()
        
        if(Draw_MV_BOXES):
        
            if(dbgFrame is None):
                image = cvDrawBoxesFromMVList(MVBoxes, image,COLOR)
            
            else:
                image = cvDrawBoxesFromMVList(MVBoxes, dbgFrame,COLOR)
        else:
            pass
###################################################################################################################
        
        ##########################################
        
        
        #Calculate center drift between detections and MVboxes
        
        #PAIR_MV_YOLO()
        
        
        #pdb.set_trace()
        if (CALC_MV_YOLO_CENTER_DRIFT):
            matches, image = MatchMVboxToYoloNew(detections,MVBoxes, image)
        
            image = DrawMatchesDiffColors(matches, detections, image)
        
            #CALC_DISTANCES()
            distances, totalError = CalcDistances(matches)
            cumulitiveDrift +=totalError
        
        
        ##########################################
        
        #print(1/(time.time()-prev_time))
        
            
        if(PRINT_DRIFT_CALC):
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
        
        cv2.line(image, (0, int(400)),(image.shape[0], int(400) ), (255, 0, 0), 1, 1)
        cv2.line(image, (0, int(200)),(image.shape[0], int(200) ), (0, 255, 0), 1, 1)
        cv2.line(image, (0, int(100)),(image.shape[0], int(100) ), (0, 0, 255), 1, 1)
        
        #print("Image is " ,image.shape[0] ," by " , image.shape[1])
        
        if(addedToFrame):   
            #pdb.set_trace()
            pass
        cv2.imshow('Frame', image)
        
        
        
##################################################################################################################################

        #Do map testing down here
        
        
        
        
        
        
        
        
        
        
##################################################################################################################################        
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        #cv2.waitKey(3)
        if(cv2.waitKey(3) & 0xFF == ord('q') or loops > breakAt):
            break
    
    if (CALC_MV_YOLO_CENTER_DRIFT):
        plt.plot(loopsArr, driftArr, label='Drift Values')
    #    plt.legend()
        plt.show()
    
    if (PRINT_FRAMERATE):
        plt.plot(loopsArr, frameRateArr, label='Framerate Values')
        plt.legend()
        plt.show()
    
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FixVSPath()
    color = np.random.randint(0,255,(DetectionPoints,3))
    import dark_motion as darknet
    YOLO()
