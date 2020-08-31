import math
import pdb
import matplotlib.pyplot as plt

ReadIn_centers = ".\OTB_OUT\\0otb_prec_plot.txt"

ReadIn_IOU = ".\OTB_OUT\\0otb_succes_plot.txt"


file_centers = open(ReadIn_centers, 'r')
lines_centers = file_centers.readlines()


file_IOU = open(ReadIn_IOU, 'r')
lines_IOU = file_IOU.readlines()

distThreshList = [0,5,10,15,20,25,30,35,40,45,50]

iouThreshList = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

#distThreshList = distThreshList.reverse()

x_cent = []
y_cent =[]

x_iou = []
y_iou =[]


for curThresh in reversed(distThreshList):
    
    TotalPoints = 0
    sumSuccess = 0
    for line in lines_centers:
        #print(line)
        line_split = line.split(" ")
        #print(line_split)
        #For each distance threshold
        TotalPoints +=1
        pixDist = float(line_split[1])
        if(pixDist<=curThresh):
            sumSuccess+=1
            
    percentageCalcCenters = sumSuccess/TotalPoints
    
    print("For %d thresh: %f", (curThresh, percentageCalcCenters))
    x_cent.append(curThresh)
    y_cent.append(percentageCalcCenters)
    
    
    
for curThresh in reversed(iouThreshList):
    
    TotalPoints = 0
    sumSuccess = 0
    for line in lines_IOU:
        #print(line)
        line_split = line.split(" ")
        #print(line_split)
        #For each distance threshold
        TotalPoints +=1
        iou = float(line_split[1])
        if(iou>=curThresh):
            sumSuccess+=1
            
    percentageCalcIOU = sumSuccess/TotalPoints
    
    print("For %d IOU: %f", (curThresh, percentageCalcIOU))
    x_iou.append(curThresh)
    y_iou.append(percentageCalcIOU)


    

    
fig, (axs1,axs2) = plt.subplots(2)
fig.suptitle('Precision and Success Plots')


axs1.plot(x_cent, y_cent, label='Yolov4 with redetect every 4')#, legend='Yolov4 with Yolo every 4th frame')
axs1.set_title('Precision plots of OPE for Walking2 Sequence !ONLY!')
axs1.set_xlabel('Location Error Threshold')
axs1.set_ylabel('Precision')
axs1.legend()



axs2.plot(x_iou, y_iou, label='Yolov4 with redetect every 4') #, legend='Yolov4 with Yolo every 4th frame')
axs2.set_title('Success plots of OPE for Walking2 Sequence !ONLY!')
axs2.set_xlabel('IOU Threshold')
axs2.set_ylabel('Success Rate')
axs2.legend()

#axs[1].legend()
plt.show()