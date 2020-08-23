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


def convertBackOTB(x, y, w, h):
    xmin = int(round(x))
    xmax = int(round(x + (w)))
    ymin = int(round(y))
    ymax = int(round(y + (h)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, COLOR):
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

USE_OTB = True

if(USE_OTB):
    path = ".\data\OTB_data\stationary\Walking\otb_Walking.avi"
    #path = ".\data\OTB_data\stationary\Walking2\otb_Walking2.avi"

    otb_gt_file = ".\data\OTB_data\stationary\Walking\groundtruth_rect.txt"

    #WHEN USING! make sure the path is to an otb data folder as well

    #plot otb ground truth
    SHOW_OTB_GT = True

else:
    SHOW_OTB_GT = False
    
cap = cv2.VideoCapture(path)
    
IMG_WIDTH = 416
IMG_HT = 416    

#pdb.set_trace()
    

    
if(USE_OTB):
    
    otb_width  = cap.get(3) 
    otb_height = cap.get(4) 
    otb_x_scale = ( IMG_WIDTH / otb_width)
    otb_y_scale = ( IMG_HT / otb_height )
else:
    otb_x_scale=1
    otb_y_scale=1


if(SHOW_OTB_GT):
    otb_file = open(otb_gt_file)
    

#cap.set(3, IMG_WIDTH)
#cap.set(4, IMG_HT)
#cap.set(3, 1280)
#cap.set(4, 720)
gtList = []
#out = cv2.VideoWriter( "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (IMG_WIDTH, IMG_HT))
    


while cap.isOpened():
    ##########################################
    ## OTB STUFF ##
    
    ret, image = cap.read()

    #resize to darknet preference
    image = cv2.resize(image, (IMG_WIDTH, IMG_HT), interpolation=cv2.INTER_LINEAR)
    
    if(SHOW_OTB_GT):
        readIn = otb_file.readline()
        
        readIn = readIn.split()
        
        #pdb.set_trace()
        
        x_gt = int(readIn[0])
        y_gt = int(readIn[1])
        w_gt = int(readIn[2])
        h_gt = int(readIn[3])
        
        #scale by otb to darknet
        # xmi,ymi,xma,yma = convertBack(x_gt, y_gt, w_gt, h_gt)
        
        # xmir = int( round(otb_x_scale*xmi))
        # xmar = int( round(otb_x_scale*xma))
        # ymir = int( round(otb_y_scale*ymi))
        # ymar = int( round(otb_y_scale*yma))
        
        # w_gt_r = int(round(xmar - xmir))
        # h_gt_r = int(round(ymar - ymir))
        # x_gt_r = int(round(xmir+w_gt_r/2))
        # y_gt_r = int(round(ymir+h_gt_r/2))
        
        # gt_box = "", 0, (x_gt_r, y_gt_r, w_gt_r, h_gt_r)
        
        x_gt = int( round(otb_x_scale*x_gt))
        w_gt = int( round(otb_x_scale*w_gt))
        
        y_gt = int( round(otb_y_scale*y_gt))
        h_gt = int( round(otb_y_scale*h_gt))
        
        gt_box = "", 0, (x_gt, y_gt, w_gt, h_gt)

        


        gtList.append(gt_box)

        COLOR = (255,32,64)
        
        image = cvDrawBoxes(gtList, image,COLOR)
        gtList.clear()
    ##########################################
    
    cv2.imshow('Frame', image)
    #cv2.waitKey(3)
    if(cv2.waitKey(3) & 0xFF == ord('q')):
        break
    #release memory

print("All done")


cap.release()
out.release()
cv2.destroyAllWindows()

    
        