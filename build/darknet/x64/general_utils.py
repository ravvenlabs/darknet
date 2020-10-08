def FindSpecific(detectionList, name):                    
    pop_tracker = 0
    
    det_len = len(detectionList)
    
    for detection in range(0, det_len):
        #if it isa not a person
#                    print(detectionsEveryFrame[0][0].decode())
        if(not detectionList[pop_tracker][0].decode() == 'person'):
            
            detectionList.pop(pop_tracker)
            pop_tracker-=1
        pop_tracker+=1

    return detectionList