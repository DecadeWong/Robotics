import pylab as pl
from roipoly import roipoly
import cv2
import numpy as np
import os
import pickle

#G_P = os.listdir('/Users/songxg64/PycharmProjects/ECE276Aproj_1/trainset')
Red_Class = [[], [], []]

img =cv2.imread('trainset/2.2.png')
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #converted image
pl.imshow(img2)
pl.colorbar()
pl.title("left click: line segment         right click: close region")

# let user draw first ROI
MyROI1 = roipoly(roicolor='r') #let user draw first ROI
MyROI1.displayROI() #show the margin of the mask

pl.imshow(MyROI1.getMask(img))
pl.show()

#display the masked graph
mask_graph = MyROI1.getMask(img)
#find the position index of masked graph for true
positions = np.where(mask_graph == True)

#slice the origin graph into R level, G level, and Blue level
R_img_level = img[:,:,0]
G_img_level = img[:,:,1]
B_img_level = img[:,:,2]

R_img = R_img_level[positions]
G_img = G_img_level[positions] #at green level of graph, the pixel is masked
B_img = B_img_level[positions] #at blue level of graph, the pixel is masked
print(R_img)

Red_Class[0].extend(R_img.tolist())
Red_Class[1].extend(G_img.tolist())
Red_Class[2].extend(B_img.tolist())

print(len(Red_Class))
print(len(Red_Class[0]))
print(len(Red_Class[1]))
