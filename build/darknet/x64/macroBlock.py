import numpy as np
import cv2
import sys
import pdb
import time
from datetime import datetime
from yolo_display_utils import *
from mv_utils import *

def SaveImage(img, filename):
    cv2.imwrite(filename, img) 

        

def PlotQuiver(frameIn, MBLocationsPre, xMotionVector, yMotionVector, MB_PARAM):

    (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH) = MB_PARAM

    i = 0
    for npMBSlice in MBLocationsPre:
        
        uLXCoord = npMBSlice[0]
        uLYCoord = npMBSlice[1]
            
        xDiff = xMotionVector[i]
        yDiff = yMotionVector[i]

        frameIn = cv2.line(frameIn, (uLXCoord+int(MB_SIZE/2),uLYCoord+int(MB_SIZE/2)), (uLXCoord+int(MB_SIZE/2) + xDiff,uLYCoord+int(MB_SIZE/2) + yDiff), (255,0,0), 2)
        i+=1


    return frameIn




def GetMacroBlockMotionVectorsVectorized(MB_PARAM, MB_LISTS, frame_gray, frame_gray_prev):
    (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector) = MB_LISTS
    
    (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH) = MB_PARAM
    
    
    macroBlockListPrev.clear()
    macroBlockListPrev = macroBlockListCur.copy()
    macroBlockListCur.clear()
    macroBlockAbsLocation.clear()
    #macroBlockAbsIDXs.clear()
######################GETBLOCKS##########################################
    #for blockIdx in range(0, numBlocks):
    #pdb.set_trace()
    for c in range(0,numBlocksHorz):
        x1 = c*MB_SIZE
        x2 = (c+1)*MB_SIZE 
        
        for r in range(0,numBlocksVert):
            y1 = r*MB_SIZE
            y2 = (r+1)*MB_SIZE
            
            selectedBlock = frame_gray[y1:y2, x1:x2]        
#            print(selectedBlock.shape)
            macroBlockListCur.append(selectedBlock.copy())
            #cv2.imshow('Frame',selectedBlock)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
            #make this a 1 time thing
            #macroBlockAbsIDXs.append((c,r))
            macroBlockAbsLocation.append((c*MB_SIZE,r*MB_SIZE))
 
    #npMacroBlockAbsLocation= np.array(macroBlockAbsLocation)
    macroBlockAbsLocationPre.clear()
    macroBlockAbsLocationPre = macroBlockAbsLocation.copy()
    
    #Collect motion information
    if(len(macroBlockListPrev) == len(macroBlockListCur)):
        #Search an N by N window
        
        #pdb.set_trace()
        for curBlockIdx in range(0, len(macroBlockListCur)):
        #if(True):    #print("Block: ", curBlockIdx)
            #curBlockIdx = 260
            
            bestSAD = float('inf')
            xMotionVector[curBlockIdx]=0
            yMotionVector[curBlockIdx]=0
            
            
            uLXCoord, uLYCoord = macroBlockAbsLocation[curBlockIdx];

            curBlock = macroBlockListCur[curBlockIdx]

            #For all window blocks 
            for i in range(-searchWindow, searchWindow):
                for j in range(-searchWindow, searchWindow):
            #        pass
                    SAD = 0
                    
                    
                    #Select MB from curBlock
                    #k 0 to 16
                    #l 0 to 16
                    XL = uLXCoord + i
                    XR = XL + MB_SIZE
                    YL = uLYCoord + j
                    YR = YL + MB_SIZE
                    
                    #Select MB from nextBlock
                    
                    sliceAdjust = False
                    extendTop = False
                    extendRight = False
                    extendLeft = False
                    extendBot = False
                    
                    if(XR>pixelH-1):
                        XR = pixelH-1
                        sliceAdjust = True
                        extendRight = True
                    if(XL<0):
                        XL=0
                        sliceAdjust = True
                        extendLeft = True
                            # #adjustments
                    if(YR>pixelV-1):
                        YR = pixelV-1
                        sliceAdjust = True
                        extendBot = True
                    if(YL<0):
                        YL=0
                        sliceAdjust = True
                        extendTop = True
                    
                    
                    if(sliceAdjust):
                        xAdj = XR-XL
                        yAdj = YR - YL
                        #if(xAdj==yAdj):
                        #    print(xAdj,yAdj)
                    
                    
                    sliceMBPrev = frame_gray_prev[YL:YR, XL:XR]
                    
                    
                    #pdb.set_trace()
                            #selectedpxlMB = int(curBlock[l,k])
                            #print()
                            #selectedPixelCurFrame = int(frame_gray_prev[Y,X])
                            #print()
                            #
                            #SAD = abs(selectedpxlMB - selectedPixelCurFrame) + SAD
                    #SAD = abs(selectedpxlMB - selectedPixelCurFrame) + SAD
                            
                    
                    
                    if(sliceMBPrev.shape == (16,16)):   
                        #print("16 by 16")
                       # pdb.set_trace()
                        SAD = np.sum(np.absolute(np.subtract(curBlock.astype(np.int16)  , sliceMBPrev.astype(np.int16)  )))
                        #pdb.set_trace()
                        #print(SAD)
                    else:
                    
                    
                        #print(sliceMBPrev.shape)
                    
                   #     pdb.set_trace
                    
                        if(extendTop):
                            sliceMBPrev = np.pad(sliceMBPrev, [(MB_SIZE-yAdj, 0), (0, 0)], mode='edge')
                        
                        if(extendBot):
                            sliceMBPrev = np.pad(sliceMBPrev, [(0, MB_SIZE-yAdj), (0, 0)], mode='edge')
                        
                        if(extendLeft):
                            sliceMBPrev = np.pad(sliceMBPrev, [(0, 0), (0, MB_SIZE-xAdj)], mode='edge')
                        
                        if(extendRight):
                            sliceMBPrev = np.pad(sliceMBPrev, [(0, 0), (MB_SIZE-xAdj, 0)], mode='edge')

                        SAD = np.sum(np.absolute(np.subtract(curBlock.astype(np.int16)  , sliceMBPrev.astype(np.int16)  )))
                       
                    
                    
                    if (SAD < bestSAD):
                        bestSAD = SAD
                        xMotionVector[curBlockIdx] = -i
                        yMotionVector[curBlockIdx] = -j
 
            
    MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector)
    #pdb.set_trace()
    return (xMotionVector, yMotionVector, MB_LISTS)

####


def GetMacroBlockMotionVectors(MB_PARAM, MB_LISTS, frame_gray, frame_gray_prev): #, macroBlockListPrev, macroBlockListCur,macroBlockAbsLocationPre, xMotionVector, yMotionVector):
    print("Macroblock motion vector call")
    (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector) = MB_LISTS
    
    (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH) = MB_PARAM
    
    macroBlockListPrev.clear()
    macroBlockListPrev = macroBlockListCur.copy()
    macroBlockListCur.clear()
    macroBlockAbsLocation.clear()
    #macroBlockAbsIDXs.clear()
######################GETBLOCKS##########################################
    #for blockIdx in range(0, numBlocks):
    for c in range(0,numBlocksHorz):
        x1 = c*MB_SIZE
        x2 = (c+1)*MB_SIZE 
        
        for r in range(0,numBlocksVert):
            y1 = r*MB_SIZE
            y2 = (r+1)*MB_SIZE
            
            selectedBlock = frame_gray[y1:y2, x1:x2]        
#            print(selectedBlock.shape)
            macroBlockListCur.append(selectedBlock.copy())
            #cv2.imshow('Frame',selectedBlock)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
            #make this a 1 time thing
            #macroBlockAbsIDXs.append((c,r))
            macroBlockAbsLocation.append((c*MB_SIZE,r*MB_SIZE))
 
    #npMacroBlockAbsLocation= np.array(macroBlockAbsLocation)
    macroBlockAbsLocationPre.clear()
    macroBlockAbsLocationPre = macroBlockAbsLocation.copy()
    
    #Collect motion information
    if(len(macroBlockListPrev) == len(macroBlockListCur)):
        #Search an N by N window
        
        
        for curBlockIdx in range(0, len(macroBlockListCur)):
            #print("Block: ", curBlockIdx)
            
            
            bestSAD = float('inf')
            xMotionVector[curBlockIdx]=0
            yMotionVector[curBlockIdx]=0
            
            
            uLXCoord, uLYCoord = macroBlockAbsLocation[curBlockIdx];

            curBlock = macroBlockListCur[curBlockIdx]

            #For all window blocks 
            for i in range(-searchWindow, searchWindow):
                for j in range(-searchWindow, searchWindow):
            #        pass
                    SAD = 0
                    
                    
                    
                    
                    
                    #For all pixels in window block
                    for k in range(0,MB_SIZE-1):
                        for l in range(0,MB_SIZE-1):
                            X = uLXCoord + k + i;
                            Y = uLYCoord + l + j;
       
                            #adjustments
                            if(X>pixelH-1):
                                X = pixelH-1
                            elif(X<0):
                                X=0

                            #adjustments
                            if(Y>pixelV-1):
                                Y = pixelV-1
                            elif(Y<0):
                                Y=0


                            #pdb.set_trace()
                            selectedpxlMB = int(curBlock[l,k])
                            #print()
                            selectedPixelCurFrame = int(frame_gray_prev[Y,X])
                            #print()
                            #
                            SAD = abs(selectedpxlMB - selectedPixelCurFrame) + SAD
                            #print(SAD)
                    
                    #if(uLXCoord+i>0 and uLYCoord+j>0):
                    
                    #    frame_gray_store = frame_gray.copy()
                    #    frame = cv2.rectangle(frame_gray, (uLXCoord+i,uLYCoord+j),  (uLXCoord+i+16,uLYCoord+j+16), (0,0,0), 2)
                    #    frame = cv2.putText(frame, str(SAD), (150, 150), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 0)) 
                    #    cv2.imshow('Frame',frame)
                    #    if cv2.waitKey(25) & 0xFF == ord('q'): 
                    #        break
                    #    frame_gray = frame_gray_store.copy()
                    
                    
                    if (SAD < bestSAD):
                        bestSAD = SAD
                        xMotionVector[curBlockIdx] = -i
                        yMotionVector[curBlockIdx] = -j
 

    MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector)
    return (xMotionVector, yMotionVector, MB_LISTS)

####


###MB update MVBOXES
def UpdateMvBoxesMacroBlock(mv_boxes, MB_PARAM, MB_LISTS, detections, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y):
    
    #global xMotionVector
    #global yMotionVector
    (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector) = MB_LISTS
    (searchWindow, numBlocksVert, numBlocksHorz, numBlocks, MB_SIZE, pixels, pixelV, pixelH) = MB_PARAM
    
    #Frame update function
    #Only do stuff if we have 2 populated MB sets
    if(len(macroBlockListPrev) == len(macroBlockListCur)):
           
        ###########APPLY MVS to boxes
            
        for yolo_det in detections:
            #unpack detection
            
            name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo_det
            #pdb.set_trace()
            xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = convertBack(
                    float(x_yolo), float(y_yolo), float(w_yolo), float(h_yolo))
        
            #Yolo corners
            #get corner points and put in tuple
            pt1_yolo = (xmin_yolo, ymin_yolo)
            pt2_yolo = (xmax_yolo, ymax_yolo)

            rect_YOLO = (pt1_yolo,pt2_yolo)
            
            total_x_mv=0
            total_y_mv=0
            contributingBoxesx=0
            contributingBoxesy=0
            
            
            #pdb.set_trace()
            for mb_idx in range(0, len(macroBlockAbsLocationPre)):
                
                #Get mb absolute location
                
                (MB_r, MB_c ) = macroBlockAbsLocationPre[mb_idx]
                
                pt1_MB = (MB_r, MB_c )
                pt2_MB = (MB_r+MB_SIZE, MB_c+MB_SIZE)
                
                rect_MB = (pt1_MB, pt2_MB)
                
                
                if(rectOverlap(rect_YOLO, rect_MB, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y)):
                   
                    #if overlap apply motion vectors
                    #frame = cv2.rectangle(frame, pt1_MB, pt2_MB, (0,0,255), 1)
                    #mb_inbounds +=1
                
                    x_mv = xMotionVector[mb_idx]
                    y_mv = yMotionVector[mb_idx]

                    #if one is nonzero
                    if(not x_mv==0):
                        contributingBoxesx+=1
                    #if one is nonzero
                    if(not y_mv==0):
                        contributingBoxesy+=1

                    total_x_mv += x_mv
                    total_y_mv += y_mv

            if(not contributingBoxesx == 0):
                #pdb.set_trace()
                total_x_mv = total_x_mv/contributingBoxesx
                
            if(not contributingBoxesy == 0):
                #pdb.set_trace()
                total_y_mv = total_y_mv/contributingBoxesy
            
            
                
                        
            
                    
            mv_box_new = name_yolo, detsProb_yolo, (x_yolo+total_x_mv,y_yolo+total_y_mv,w_yolo,h_yolo)
            
            mv_boxes.append(mv_box_new)
            
    
    MB_LISTS = (macroBlockListPrev, macroBlockListCur, macroBlockAbsLocationPre, macroBlockAbsLocation, xMotionVector, yMotionVector)
    return mv_boxes, MB_LISTS
###