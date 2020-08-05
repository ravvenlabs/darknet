from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import sys
#import darknet

import pdb
#openCV static info
path = "./data/test.mp4"


####################################################################

#Frames till next optical flow point refresh
OF_DET_SKIP = 6

#frames till next yolo call 
YOLO_DET_SKIP = 15



#Rectangles for box movement and crosshairs on MV in said rectangle (And print loactions of each)
DEBUG_OBJECTS = False

#MV inside of rectangle buffer for outside of rectangle
MV_RECT_BUFFER_VERT = 10
MV_RECT_BUFFER_HORZ = 0


#Yolo accuracy required to make a bbox
YOLO_DET_THRESH = .33

#Insert a delay from one frame to the next
SLOW_MODE =False
#with VOC
#YOLO_DET_THRESH = .10

#put circlews on most recent O.F. point
CV_CIRCLE_ON = False

#more objects detected and tracked
DetectionPoints = 200

#draw yolo bboxes (every frame green)
drawYOLO = False

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
        
        throw, detection, out = detection 
    
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
        #cv2.putText(img,
        #            detection[0].decode() +
        #            " [" + str(round(detection[1] * 100, 2)) + "]",
        #            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            [red,green,blue], 2)
    return img


netMain = None
metaMain = None
altNames = None

def cvDrawOneBox(detections, img, COLOR):

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
        cv2.rectangle(img, pt1, pt2, (red, green, blue), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        break
    return img


netMain = None
metaMain = None
altNames = None

def AssocDetections(detections):
    
    MV_and_Detections = []
    
    id = 0
    for detection in detections:
        MV_and_Detections.append( (id, detection, list()) ) 
        id+=1
        
    return MV_and_Detections

def InsideRect(a,b,x,y,w,h, bufferV,bufferH):
    
    return (a>=(x-bufferH) and a<=(x+w+bufferH) and b>=(y-bufferV) and b<=(y+h+bufferV))
    
    
def UpdateMvBoxes(detections, newFramePoints, oldFramePoints, dbgFrame=None, mask = None):
    
    element = 0
    addedToFrame = False
    for packed in detections:
        
        (id, detection, mvs) = packed
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
    
            DistY = DistY / total
            DistY = DistY / total
                
        if(DistX != 0 or DistY != 0):
            #pdb.set_trace()
            NewDetection = MoveDetection(detection, DistX, DistY)
        
            repacked = (id, NewDetection, mvs)
        
            detections[element] = repacked
        
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
    cap = cv2.VideoCapture("./data/test.mp4")
    
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
    while cap.isOpened():
        addedToFrame = False
        loops=loops+1
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
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                if(CV_CIRCLE_ON):
                    frame_read = cv2.circle(frame_read,(a,b),5,color[i].tolist(),-1)
               # if(a>maxx):
                #    maxx=a
             #       print("MAX X" , maxx)
              #      print("MAX Y", maxy)
                    
               # if(c>maxx):
                #    maxx=c
               #     print("MAX X" , maxx)
                #    print("MAX Y", maxy)
                    
               # if(b>maxy):
                #    maxy=b
                 #   print("MAX X" , maxx)
                  #  print("MAX Y", maxy)
                    
               # if(d>maxy):
               #     maxy=d
                   # print("MAX X" , maxx)
                    #print("MAX Y", maxy)
        image = cv2.add(frame_read,mask)
       
        #cv2.imshow('Demo', image)
        #################################################
        
        frame_resized = cv2.resize(image,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            
        if(loops%YOLO_DET_SKIP==0 or detections is None):
                                                            #IT WAS: thresh=0.25
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=YOLO_DET_THRESH)
        #image = cvDrawOneBox(detections, frame_resized)
        
        COLOR = (0,255,0)
        
        if(drawYOLO):
            image = cvDrawBoxes(detections, frame_resized,COLOR)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
###################################################################################################################
        
        #(id, detection, [])
        
        
        if(loops%YOLO_DET_SKIP==0 or MVBoxes is None):
            MVBoxes = AssocDetections(detections)
            #sys.stdout.write("MVBoxes assigned\n")
        
        else:
        
            MVBoxes, dbgFrame,addedToFrame = UpdateMvBoxes(MVBoxes, good_new, good_old, image, mask)
        COLOR = (0,0,255)
        #pdb.set_trace()
        Draw = True
        if(Draw):
        
            if(dbgFrame is None):
                image = cvDrawBoxesFromMVList(MVBoxes, image,COLOR)
            
            else:
                image = cvDrawBoxesFromMVList(MVBoxes, dbgFrame,COLOR)
        else:
            pass
###################################################################################################################
        
        
        #print(1/(time.time()-prev_time))
        
        sys.stdout.write("Frame Rate: %f     \r" % (1/(time.time()-prev_time)) )
        sys.stdout.flush()
        
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
        if(cv2.waitKey(3) & 0xFF == ord('q')):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FixVSPath()
    color = np.random.randint(0,255,(DetectionPoints,3))
    import dark_motion as darknet
    YOLO()
