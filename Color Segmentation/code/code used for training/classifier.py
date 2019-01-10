#prior probability of class
from my_funct import cov_matrix, G_probability
import pickle
import numpy as np
import cv2
import pylab as pl

redclass_obj = open('redclass.pkl', 'rb') # this file save a list for red class, which includes R G B value for that
red_data = pickle.load (redclass_obj)
redclass_obj.close()

brownclass_obj = open('brown.pkl', 'rb') # this file save a list for red class, which includes R G B value for that
brown_data = pickle.load (brownclass_obj)
brownclass_obj.close()

nsredclass_obj = open('notredbarrel.pkl', 'rb') # this file save a list for red class, which includes R G B value for that
nsred_data = pickle.load (nsredclass_obj)
nsredclass_obj.close()

yellowclass_obj = open('yellow.pkl', 'rb') # this file save a list for red class, which includes R G B value for that
yellow_data = pickle.load (yellowclass_obj)
yellowclass_obj.close()

redarea_obj = open('redarea.pkl', 'rb') # this file save a list for red class, which includes R G B value for that
redarea_data = pickle.load (redarea_obj)
redarea_obj.close()


[cov_red, mean_red] = cov_matrix(red_data)
print(cov_red, mean_red)
[cov_nsred, mean_nsred] = cov_matrix(nsred_data)
print(cov_nsred, mean_nsred)
[cov_brown, mean_brown] = cov_matrix(brown_data)
print(cov_brown, mean_brown)
[cov_yellow, mean_yellow] = cov_matrix(yellow_data)
print(cov_yellow, mean_yellow)


img =cv2.imread('testset/006.JPG')
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
A = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
img3 = A[::10,::10,:]


dimension = img3[:,:,0].shape #where dimension[0] is number of row, and dimension[1] is number of column
classified_picture = np.zeros((img3.shape[0], img3.shape[1], img3.shape[2]))

for i in range(dimension[0]):
    for j in range(dimension[1]):

        B_RGB = [[img3[i,j,0]], [img3[i,j,1]], [img3[i,j,2]]]

        RGB = np.matrix(B_RGB)

        PYred_XRGB = G_probability (mean_red, cov_red, RGB)
        PYnsred_XRGB = G_probability (mean_nsred, cov_nsred, RGB)
        PYbrown_XRGB = G_probability (mean_brown, cov_brown, RGB)
        PYyellow_XRGB = G_probability (mean_yellow, cov_yellow, RGB)

        #check PYred_XRGB = PXRGB_Yred * PYred
        #PYred_XRGB = PXRGB_Yred
        #PYnsred_XRGB = PXRGB_Ynsred
        #PYbrown_XRGB = PXRGB_Ybrown
        #PYyellow_XRGB = PXRGB_Yyellow


        if max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYred_XRGB:
            #assign this is red class
            classified_picture[i, j] = 0
            #classified_picture[i, j, 0] = 0
            #classified_picture[i, j, 1] = 0
            #classified_picture[i, j, 2] = 0

        elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYnsred_XRGB:
            #assign this pixel is not so red class
            classified_picture[i, j] = 1
            #classified_picture[i, j, 0] = 0
            #classified_picture[i, j, 1] = 0
            #classified_picture[i, j, 2] = 0
        elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYbrown_XRGB:
            classified_picture[i, j] = 1
            #classified_picture[i, j, 0] = 0
            #classified_picture[i, j, 1] = 0
            #classified_picture[i, j, 2] = 0
        elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYyellow_XRGB:
            #assign this pixel is yellow class
            classified_picture[i, j] = 1
            #classified_picture[i, j, 0] = 0
            #classified_picture[i, j, 1] = 0
            #classified_picture[i, j, 2] = 0


pl.imshow(classified_picture)
pl.show()









