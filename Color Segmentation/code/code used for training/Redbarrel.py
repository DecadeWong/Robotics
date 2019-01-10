import pylab as pl
from roipoly import roipoly
import cv2
import numpy as np
import os
import pickle
from skimage import measure

G_P = os.listdir('/Users/songxg64/PycharmProjects/ECE276Aproj_1/trainset') #get group of picture name(list)
Red_Class = [[], [], []] # initialize the red class as a list which contains three sub list for RGB
Red_Class_area = []

for img_file in G_P:
    img =cv2.imread('trainset/'+ img_file)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #converted image


    pl.imshow(img2) #show the converted image
    pl.colorbar() #show the color bar on the side of image
    pl.title("left click: line segment         right click: close region")

    # let user draw first ROI in red color
    MyROI1 = roipoly(roicolor='r') #let user draw first ROI
    MyROI1.displayROI() #show the margin of the mask
    pl.imshow(MyROI1.getMask(img2)) #show the mask graph, only two color in this graph, intermediate
    pl.show()#keep show the image, final decided

    mask_graph = MyROI1.getMask(img) #display the masked graph
    positions = np.where(mask_graph == True) #find the position index of masked graph for true

    img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
    R_img_level = img3[:,:,0] #slice the origin graph into R level
    G_img_level = img3[:,:,1] #slice the origin graph into G level
    B_img_level = img3[:,:,2] #slice the origin graph into B level

    R_img = R_img_level[positions]
    G_img = G_img_level[positions] #at green level of graph, the pixel is masked
    B_img = B_img_level[positions] #at blue level of graph, the pixel is masked

    Red_Class[0].extend(R_img.tolist()) #append final data R data list in red class(list) at first space
    Red_Class[1].extend(G_img.tolist()) #append final data G data list in red class(list) at second space
    Red_Class[2].extend(B_img.tolist()) #append final data B data list in red class(list) at third space

    label_mask = measure.label(mask_graph)
    Gproperty = measure.regionprops(label_mask) # the property of mask graph
    Red_Class_area.append(Gproperty[0].area)

output_redclass = open('redclass.pkl','wb') # this file save a list for red class, which includes R G B value for that
pickle.dump (Red_Class, output_redclass)
output_redclass.close()

output_redarea = open('redarea.pkl','wb')
pickle.dump (Red_Class_area, output_redarea)
output_redarea.close()


