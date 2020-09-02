#simulator
import pdb

frameBuffer = [None]*10
DisplayBuffer = []

MVBuffer = [None]*10


#Doesnt display last 10 frames
frames = 40

FRAME_SKIP = 10

def displayFrame(frameContent):
    print("Frame: ", frameContent)

def calcMVs(curFrame):
    return "MV"+str(curFrame)+" to "+str(curFrame+1)

for frameIndex in range(0,frames):
    
    curFrame=frameIndex+1
    #Push current frame to frame stack for display
    #
    if(frameIndex>9):
        ##We have full buffers, dsiplay frames
        print("At index: ", frameIndex, "Display: ", DisplayBuffer[frameIndex%FRAME_SKIP])
    
    
    MVBuffer[frameIndex%FRAME_SKIP] = calcMVs(frameIndex)
    frameBuffer[frameIndex%FRAME_SKIP] = frameIndex
    

    #If we are not at a yolo result frame
    if(not curFrame%FRAME_SKIP==0):
        ##Calulate MVs
        pass
        
    if(curFrame%FRAME_SKIP==0):
        print("Yolo finished at frame index: "+str(frameIndex))
        print("Motion vector past ", FRAME_SKIP," frames")
        print(MVBuffer)
        print("Frames to show past ", FRAME_SKIP," frames")
        print(frameBuffer)
        #Apply motion vectors to Detections
        
        MVBuffer = [None]*10
        DisplayBuffer = frameBuffer.copy()
        #MVBuffer.clear()
    
    #There is a 10 frame delay
    
    
    
    
    #displayFrame(curFrame)
