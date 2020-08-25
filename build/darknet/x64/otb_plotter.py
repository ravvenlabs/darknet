import math
import pdb
import matplotlib.pyplot as plt

ReadIn = ".\OTB_OUT\\0otb_prec_plot.txt"

file = open(ReadIn, 'r')
lines = file.readlines()

distThreshList = [0,5,10,15,20,25,30,35,40,45,50]

#distThreshList = distThreshList.reverse()

x = []
y=[]

for curThresh in reversed(distThreshList):
    
    TotalPoints = 0

    sumSuccess = 0
    
    for line in lines:
        #print(line)
        line_split = line.split(" ")
        #print(line_split)
    
    
        #For each distance threshold
        TotalPoints +=1
        pixDist = float(line_split[1])
        if(pixDist<=curThresh):
            sumSuccess+=1
            
    percentageCalc = sumSuccess/TotalPoints
    

    
    print("For %d thresh: %f", (curThresh, percentageCalc))
    x.append(curThresh)
    y.append(percentageCalc)
plt.plot(x, y, label='Precision plots of OPE for Walking2 Sequence !ONLY!')
plt.legend()
plt.show()