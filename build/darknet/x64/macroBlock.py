import numpy as np
import cv2
import sys
import pdb
import time
from datetime import datetime
import yolo_display_utils

#pdb.set_trace()

path = "./data/test.mp4"
#path = "./data/aerial.mp4"

cap = cv2.VideoCapture(path)
cap.set(3, 416)
cap.set(4, 416)
if(cap.isOpened()==False):
    print("Error opening")


pixelH = 416
pixelV = 416

#pixelH = 104
#pixelV = 104


pixels=pixelH*pixelV
#MB_SIZE = 13
MB_SIZE = 16

#MB_SIZE = 208

rowIdx = 0
colIdx = 0
rowsCounted = 0

numBlocks = int(pixels / (MB_SIZE*MB_SIZE))
numBlocksHorz = int(pixelH/MB_SIZE)
numBlocksVert = int(pixelV/MB_SIZE)

#numBlocksHorz = 2
#numBlocksVert = 2

macroBlockListPrev = []

macroBlockListCur = []
macroBlockAbsLocation = []
macroBlockAbsIDXs = []
searchWindow = 10
searchWindow = 3

xMotionVector = [None]* numBlocks
yMotionVector = [None]*numBlocks

#print( )
bestSAD = None


print("Macroblocks: ", numBlocks)
# Create some random colors


loops = 1


ret, old_frame = cap.read()

old_frame = cv2.resize(old_frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
        
frame = None

ret, frame = cap.read()  
frame = cv2.resize(frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
now =  datetime.now()
#For current frame
while(cap.isOpened()):


    sys.stdout.write("Frame %d     \r" % (loops) )
    sys.stdout.flush()
   
    ret, frame = cap.read()
    
    frame_gray_prev = frame_gray.copy()
    frame_prev = frame.copy()
    
    frame = cv2.resize(frame, (pixelH, pixelV), interpolation=cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    prev = now
    
    now = datetime.now()
    print((now-prev).total_seconds(), " Seconds elapsed!")
    
    #partition frame 1 into 16x16 macroblocks
    macroBlockListPrev = macroBlockListCur.copy()
    macroBlockListCur.clear()
    macroBlockAbsLocation.clear()
    macroBlockAbsIDXs.clear()
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
            macroBlockAbsIDXs.append((c,r))
            macroBlockAbsLocation.append((c*MB_SIZE,r*MB_SIZE))
   #         cv2.imshow('Frame',selectedBlock)
   #         if cv2.waitKey(25) & 0xFF == ord('q'): 
  #              break
 #           pdb.set_trace()

######################GETBLOCKS##########################################

    npMacroBlockAbsLocation= np.array(macroBlockAbsLocation)
    
    #Collect motion information
    if(len(macroBlockListPrev) == len(macroBlockListCur)):
        #Search an N by N window
        
        
        for curBlockIdx in range(0, len(macroBlockListCur)):
            #print("Block: ", curBlockIdx)
            
            
            bestSAD = float('inf')
            xMotionVector[curBlockIdx]=0
            yMotionVector[curBlockIdx]=0
            
            
            uLXCoord, uLYCoord = macroBlockAbsLocation[curBlockIdx];
            #frame = cv2.rectangle(frame, (uLXCoord,uLYCoord), (uLXCoord+MB_SIZE,uLYCoord+MB_SIZE), (0,255,0), 1)
            #print(uLXCoord,uLYCoord)
            #time.sleep(0.05)
            
            #print(uLXCoord, uLYCoord)
            #pdb.set_trace()
        
            curBlock = macroBlockListCur[curBlockIdx]
            #print(curBlock)
            #cv2.imshow('Frame',curBlock)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
            
            #pdb.set_trace()
        
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
                    
                    if (SAD < bestSAD):
                        bestSAD = SAD
                        xMotionVector[curBlockIdx] = -i
                        yMotionVector[curBlockIdx] = -j
                  
        #exit()                 
                            #exit()
#                            SAD = abs()
                            
    
    #Show motion information
            #Index frame experiment
            #for i in range(0,416):
            #    frame[40,i]=(255,255,255)
            
            #cv2.imshow('Frame',frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
        #pdb.set_trace()
        #Apply motion vectors
        
        
        for npMBSlice in npMacroBlockAbsLocation:
            uLXCoord = npMBSlice[0]
            uLYCoord = npMBSlice[1]
            #frame = cv2.rectangle(frame, (uLXCoord,uLYCoord), (uLXCoord+MB_SIZE,uLYCoord+MB_SIZE), (0,255,0), 1)
            #cv2.imshow('Frame',frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
        
        
        
        npxMotionVector = np.array(xMotionVector)
        npyMotionVector = np.array(yMotionVector)

        npMacroBlockAbsLocationPre = np.copy(npMacroBlockAbsLocation)
        
        
        #Add X
        npMacroBlockAbsLocation[:,0] = np.add(npMacroBlockAbsLocation[:,0],npxMotionVector)
        
        #Add Y
        npMacroBlockAbsLocation[:,1] = np.add(npMacroBlockAbsLocation[:,1], npyMotionVector)

        npMacroBlockAbsLocationDiff = np.equal(npMacroBlockAbsLocation,npMacroBlockAbsLocationPre)
            
        npMacroBlockAbsLocationDiff = np.logical_and(npMacroBlockAbsLocationDiff[:,0],npMacroBlockAbsLocationDiff[:,1])
            
 #       npMacroBlockListCur
        #xComp = xMotionVector[curMotIdx]
        #yComp = yMotionVector[curMotIdx]
        #pdb.set_trace()
        keepIdx = 0
        for npMBSlice in npMacroBlockAbsLocation:
            uLXCoord = npMBSlice[0]
            uLYCoord = npMBSlice[1]
            
            if(npMacroBlockAbsLocationDiff[keepIdx]==False):
            
                frame = cv2.rectangle(frame, (uLXCoord,uLYCoord), (uLXCoord+MB_SIZE,uLYCoord+MB_SIZE), (255,0,0), 1)
            #cv2.imshow('Frame',frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'): 
            #    break
            keepIdx+=1
 
#        for npMBIDX in npMacroBlockAbsLocation:
#            print(npMBIDX)
    #    cv2.imshow('Frame',macroBlockListCur[i])
    #    if cv2.waitKey(25) & 0xFF == ord('q'): 
    #        break
    #    pdb.set_trace()
        
        
        
    if(ret==True):
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else:
        break
    
 
    
    loops+=1


cap.release()
cv2.destroyAllWindows()