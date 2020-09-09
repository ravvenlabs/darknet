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


COLOR_blue = (255,0,0)
COLOR_green = (0,255,0)
COLOR_red = (0,0,255)
COLOR_yellow = (0,255,255)
COLOR_white = (255,255,255)


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
    
def cvDrawCenters(frameList, img, COLOR,size=1):
    red,green,blue = COLOR
    
    for detections in frameList:
        
        for detection in detections:
            


            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            
            pt1 = (int(round(x)), int(round(y)))
            
            cv2.circle(img,pt1,size,COLOR,-1)

            
            #cv2.rectangle(img, pt1, pt2, (red,green,blue), 1)
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
    
    
    
#This function draws a white line between 2 boxes
def cvDrawLinkCenter(mv,yolo, image):
    
    name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mv
        
    name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo
    
    image = cv2.line(image, (int(x_mv),int(y_mv)),(int(x_yolo),int(y_yolo)), (255,255,255), 2)
    
    
    return image

#This function will go through a list of pairs and draw the connecting line.
#It will also show the paired boxes in different colors for easy inspection of pairing
def DrawMatchesDiffColors(matches, detections, image, link_only=False):
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
        if(not link_only):
            image = cvDrawOneBox(box1, image, COLOR)
            
            image = cvDrawOneBox(box2, image, COLOR)
        
    return image