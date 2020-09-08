#simulator
import pdb

frameBuffer = [None]*10
DisplayBuffer = [None]*10

MVBuffer = [None]*10


#Doesnt display last 10 frames
frames = 40

YOLO_DET_SKIP = 10

def displayFrame(frameContent):
    print("Frame: ", frameContent)

def calcMVs(curFrame):
    return "MV"+str(curFrame)+" to "+str(curFrame+1)
lastYolo = 0
for frameIndex in range(0,frames):
    
    
    curFrame=frameIndex+1
    #Push current frame to frame stack for display
    #
    if((curFrame-1)>9):
        ##We have full buffers, dsiplay frames
        print("At index: ", (curFrame-1), "Display: ", DisplayBuffer[(curFrame-1)%YOLO_DET_SKIP])
        #print(DisplayBuffer)
    
    
    #frameBuffer[(curFrame-1)%YOLO_DET_SKIP] = (curFrame-1)
    

    #If we are not at a yolo result frame
    if(not curFrame%YOLO_DET_SKIP==0):
        ##Calulate MVs
        MVBuffer[(curFrame-1)%YOLO_DET_SKIP] = calcMVs((curFrame-1))
        
    if(curFrame%YOLO_DET_SKIP==0):
        print("Yolo finished at frame index: "+str((curFrame-1)))
        
        
        print("Motion vector past ", YOLO_DET_SKIP," frames")
        print(MVBuffer)
        print("Frames to show past ", YOLO_DET_SKIP," frames")
        #print(frameBuffer)
        #Apply motion vectors to Detections
        
        for bufIdx in range(0, YOLO_DET_SKIP):
            #print(bufIdx)
            #apply mot vec[i]
            DisplayBuffer[bufIdx]=str(lastYolo) + " dets and move with " + str(MVBuffer[bufIdx])
        
        DisplayBuffer[YOLO_DET_SKIP-1] = "Yolo from frame " + str(curFrame - YOLO_DET_SKIP)
        
        MVBuffer = [None]*10
        #DisplayBuffer #= frameBuffer.copy()
        #MVBuffer.clear()
        lastYolo = curFrame
        #start next yolo
    
    #There is a 10 frame delay
    
    
    
    
    #displayFrame(curFrame)
