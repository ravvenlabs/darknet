#simulator
import pdb

frameBuffer = [None]*10
DisplayBuffer = [None]*10

MVBuffer = [None]*10


#Doesnt display last 10 frames
frames = 40

FRAME_SKIP = 10

def displayFrame(frameContent):
    print("Frame: ", frameContent)

def calcMVs(curFrame):
    return "MV"+str(curFrame)+" to "+str(curFrame+1)
lastYolo = 0
for frameIndex in range(0,frames):
    
    
    curFrame=frameIndex+1
    #Push current frame to frame stack for display
    #
    if(frameIndex>9):
        ##We have full buffers, dsiplay frames
        print("At index: ", frameIndex, "Display: ", DisplayBuffer[frameIndex%FRAME_SKIP])
        #print(DisplayBuffer)
    
    
    frameBuffer[frameIndex%FRAME_SKIP] = frameIndex
    

    #If we are not at a yolo result frame
    if(not curFrame%FRAME_SKIP==0):
        ##Calulate MVs
        MVBuffer[frameIndex%FRAME_SKIP] = calcMVs(frameIndex)
        
    if(curFrame%FRAME_SKIP==0):
        print("Yolo finished at frame index: "+str(frameIndex))
        
        
        print("Motion vector past ", FRAME_SKIP," frames")
        print(MVBuffer)
        print("Frames to show past ", FRAME_SKIP," frames")
        print(frameBuffer)
        #Apply motion vectors to Detections
        
        for bufIdx in range(0, FRAME_SKIP):
            print(bufIdx)
            #apply mot vec[i]
            DisplayBuffer[bufIdx]=str(lastYolo) + " dets and move with " + str(MVBuffer[bufIdx])
        
        DisplayBuffer[FRAME_SKIP-1] = "Yolo from frame " + str(curFrame - FRAME_SKIP)
        
        MVBuffer = [None]*10
        #DisplayBuffer #= frameBuffer.copy()
        #MVBuffer.clear()
        lastYolo = curFrame
        #start next yolo
    
    #There is a 10 frame delay
    
    
    
    
    #displayFrame(curFrame)
