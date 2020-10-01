import cv2
import numpy as np
import glob
import pdb

#pdb.set_trace()
 
img_array = []
for filename in glob.glob('.\data\OTB_data\\noPlace\\Man\\img\*.jpg'): #glob.glob('.\data\OTB_data\\noPlace\\Crowds\\img*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('.\data\OTB_data\\noPlace\\Man\\otb_man.avi',fourcc, 30.0, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()