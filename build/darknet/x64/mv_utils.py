from yolo_display_utils import *

#Copy detections
def AssocDetections(detections):
    
    MV_and_Detections = []
    
    #id = 0
    for detection in detections:
        MV_and_Detections.append( detection) #(id, detection, list()) ) 
        #id+=1
        
    return MV_and_Detections
    
#Determine whether a point is inside a rectangle (Error buffer is taken into account too)
def InsideRect(a,b,x,y,w,h, bufferV,bufferH):
    return (a>=(x-bufferH) and a<=(x+w+bufferH) and b>=(y-bufferV) and b<=(y+h+bufferV))
    
#Move the motion vector boxes 
#takes in the new detections and the new and old optical flow points
def UpdateMvBoxes(detections, newFramePoints, oldFramePoints, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ, DEBUG_OBJECTS=False, dbgFrame=None, mask = None):
    
    element = 0
    addedToFrame = False
    
    #For each detection
    for detection in detections:
        
        #(id, detection, mvs) = packed
        #pdb.set_trace()
        #print(id)
        
        DistX = 0
        DistY = 0
        #print(detection)
        #unpack this detection
        name, detsProb, (x,y,w,h) = detection
           
        total = 0
        
        #for each set of points in new and old
        for i,(new,old) in enumerate(zip(newFramePoints, oldFramePoints)):
                
                a,b = new.ravel()
                c,d = old.ravel()
               
                #pdb.set_trace()
                
                #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                #frame_read = cv2.circle(frame_read,(a,b),5,color[i].tolist(),-1)
        
                #if new point is inside rectangle
                if(InsideRect(a,b,x,y,w,h, MV_RECT_BUFFER_VERT, MV_RECT_BUFFER_HORZ)):
                
                    total+=1
                    #mask = cv2.line(mask, (int(0),int(y)),(int(1000),int(y+h)), (0,255,150), 5)
                    #mask = cv2.line(mask, (int(x),int(0)),(int(x+w),int(1000)), (0,255,150), 5)
                    #pdb.set_trace()
                    
                    #For single frame debug stop
                    addedToFrame = True
                    
                    if(DEBUG_OBJECTS):
                        xmin, ymin, xmax, ymax = convertBack( float(x), float(y), float(w), float(h))
                        pt1 = (xmin, ymin)
                        pt2 = (xmax, ymax)
                        mask = cv2.rectangle(mask, (pt1),(pt2), (255,0,0), 5)
                        mask=cv2.line(mask, (int(a), 0),(int(a), mask.shape[0]), (255, 255, 0), 1, 1)
                        mask=cv2.line(mask, (0, int(b)),(mask.shape[1], int(b) ), (255, 255, 0), 1, 1)
#                       #pdb.set_trace()
                        print("Rectangle at " ,pt1 ," by " , pt2)
                        print("OpenCV point is at " ,a ," by " , b)
                    else:
                        pass
                
                    #Update the x and y change based on new and old point movement
                    DistX = DistX +  a-c
                    DistY = DistY + b-d
    
        #Divide distance by number of mvs in box
        if(total>0):
            DistX = DistX / total
            DistY = DistY / total
                
        #If the x or y needs to be adjusted because shift is nonzero, call move function
        if(DistX != 0 or DistY != 0):
            #pdb.set_trace()
            
            #This function actually accesses the detections properties
            NewDetection = MoveDetection(detection, DistX, DistY)
            #replace
            detections[element] = NewDetection
        element+=1
        
    return detections, dbgFrame, addedToFrame
    
    
#This function actually accesses the detections struct paramss and updates them
def MoveDetection(detection, dx,dy):
    #x -> detection[2][0]
    #y -> detection[2][1]
    name, detsProb, (x,y,w,h) = detection
    #x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
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

#This function will return true if one rectangle overlaps with another
#pass in top left and bottom right tuples of points
#top left 1, bot right 1
#top left 2, bot right 2
#rect 1 is mvbox
#rect 2 is yolo box
def rectOverlap(rect1,rect2, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y):
    
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
    
    
#This function is for matching a new detection to an existing box for an object
#It will go thru all existing boxes and then see if there are any new detection boxes that overlap the old mv box
#If there are, candidates distances are calculated and the closest box is matched to the existing box
def MatchMVboxToYoloNew(detections, mvBoxes, MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y, dbgimg=None):
    pass
    mvBoxesWithYoloID = []
    candidate = []
    dist = 10000
    
    #for motion vector boxes
    for mvbox in mvBoxes:
    
        name_mv, detsProb_mv, (x_mv,y_mv,w_mv,h_mv) = mvbox
        xmin_mv, ymin_mv, xmax_mv, ymax_mv = convertBack(
                    float(x_mv), float(y_mv), float(w_mv), float(h_mv))
        
        #get corner points and put in tuple
        pt1_mv = (xmin_mv, ymin_mv)
        pt2_mv = (xmax_mv, ymax_mv)
        count = 0
        at_least_one_match = False
    
        #For all new detections
        for yolo_det in detections: 
            #unpack
            name_yolo, detsProb_yolo, (x_yolo,y_yolo,w_yolo,h_yolo) = yolo_det
            #pdb.set_trace()
            xmin_yolo, ymin_yolo, xmax_yolo, ymax_yolo = convertBack(
                    float(x_yolo), float(y_yolo), float(w_yolo), float(h_yolo))
            
            #get corner points and put in tuple
            pt1_yolo = (xmin_yolo, ymin_yolo)
            pt2_yolo = (xmax_yolo, ymax_yolo)

            count +=1

            #If there is an overlap of an mv and a yolo box
            if(rectOverlap( (pt1_mv, pt2_mv), ( pt1_yolo, pt2_yolo), MV_YOLO_ASSOCIATION_BUFFER_X, MV_YOLO_ASSOCIATION_BUFFER_Y )):
                #sys.stdout.write("Rect overlap\r")
                #sys.stdout.flush()
                at_least_one_match = True
                #overlap found, make match
                #mvBoxesWithYoloID.append((mvbox, yolo_det))
                #break
                #calulate distance fo the boxes
                dist = DiagDist(x_mv,y_mv, x_yolo,y_yolo)              
                #add it to the list
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
            
            #for each candidate box determine which is the closest
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
            
            #Add matched pair to return list
            mvBoxesWithYoloID.append((candidate[least][0], mvbox))
            candidate.clear()

    if(dbgimg is None):
        return mvBoxesWithYoloID
    else:
        #print("Returning with dbg image")
        
        return mvBoxesWithYoloID, dbgimg
        

#This is a distance calculation between two points
def DiagDist(x1,y1,x2,y2):
    #return center point
    
    dx = x1-x2
    dy= y1-y2
    dist = math.sqrt( dx*dx+dy*dy )
    
    return dist
    
def XDist(x1,y1,x2,y2):
    #return center point
    
    dx = x1-x2
    # dy= y1-y2
    # dist = math.sqrt( dx*dx+dy*dy )
    
    return dx
    
#This is for the drift calulations. It calulates the distances between each pair of boxes
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
        #dist = XDist(x_mv,0, x_yolo,0)
        
        
        
        
        distList.append(dist)
    
        #Get yolobox diag
    
        xmin, ymin, xmax, ymax = convertBack(x_yolo,y_yolo,w_yolo,h_yolo)
        totalDriftAmount += (dist)/(DiagDist( xmin,ymin,xmax,ymax))
    
    
    
    return distList, totalDriftAmount 