from my_funct import G_probability #, linear_regression
import numpy as np
import cv2
import pylab as pl
from PIL import Image
def myAlgorithm(img):

    cov_red = [[ 983.45516294, -20.82693649, 251.59740008], [-20.82693649, 18.27097824, -39.23327015],
               [ 251.59740008, -39.23327015, 458.49355809]]
    mean_red = [[60.21777405206787], [121.75193536269101], [187.3117150894875]]

    cov_nsred = [[ 1995.54352586, -145.25809121, -111.5877214 ], [ -145.25809121, 83.23549837, -117.67712958],
                 [ -111.5877214, -117.67712958, 493.246937  ]]
    mean_nsred = [[94.15792807168283], [115.255097581932], [172.50993494537866]]

    cov_brown = [[ 1714.41542918, -125.00920446, 382.1762053 ], [ -125.00920446, 83.38935013, -121.41269209],
                 [  382.1762053, -121.41269209, 269.16857731]]
    mean_brown = [[75.56044612666769], [114.84189721740381], [153.4726336741301]]

    cov_yellow = [[ 1907.97039881, -462.80228832, 144.84046221], [ -462.80228832, 249.39126596, -114.30270918],
                  [  144.84046221, -114.30270918, 105.07373941]]
    mean_yellow = [[120.61519891291778], [96.64161703480976], [153.83091679195095]]


    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    A = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
    img3 = A[::10,::10,:]
    img4 = cv2.cvtColor(img3, cv2.COLOR_YCR_CB2BGR)
    distance =[10, 10, 14, 2, 2, 2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,8,8,8,8,9,9]
    #param = linear_regression(red_area, distance)
    param = [-9.43267079e-05, 6.83717067e+00]
    #print(param)

    dimension = img3[:,:,0].shape #where dimension[0] is number of row, and dimension[1] is number of column

    classified_picture = np.zeros((img3.shape[0], img3.shape[1], img3.shape[2]), dtype=np.uint8)

    for i in range(dimension[0]):
        for j in range(dimension[1]):
            B_RGB = [[img3[i,j,0]], [img3[i,j,1]], [img3[i,j,2]]]

            RGB = np.matrix(B_RGB)
            #check PYred_XRGB = PXRGB_Yred * PYred
            PYred_XRGB = G_probability (mean_red, cov_red, RGB)
            PYnsred_XRGB = G_probability (mean_nsred, cov_nsred, RGB)
            PYbrown_XRGB = G_probability (mean_brown, cov_brown, RGB)
            PYyellow_XRGB = G_probability (mean_yellow, cov_yellow, RGB)

            if max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYred_XRGB:
                #assign this is red class
                #classified_picture[i, j] = 0
                classified_picture[i, j] = [255, 0, 0]

            elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYnsred_XRGB:
                #assign this pixel is not so red class
                #classified_picture[i, j] = 1
                classified_picture[i, j] = [255, 255, 255]

            elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYbrown_XRGB:
                classified_picture[i, j] = [255, 255, 255]
                #classified_picture[i, j] = 1
            elif max(PYred_XRGB, PYnsred_XRGB, PYbrown_XRGB,PYyellow_XRGB ) == PYyellow_XRGB:
                #assign this pixel is yellow class
                classified_picture[i, j] = [255, 255, 255]
                #classified_picture[i, j] = 1


    img_final = Image.fromarray(classified_picture, 'RGB')
    #img.show()
    k = np.array(img_final)

    imgray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    #find the contour area
    max_value = max(area)
    max_index = area.index(max_value)

    del area[max_index]
    del contours[max_index]

    final_area = []
    final_contours = []
    for i in range(len(contours)):
        if area[i] > max(area)/10:
            final_area.append(area[i])
            final_contours.append(contours[i])


    max_final_area = max(final_area)
    max_final_area_index = final_area.index(max_final_area)
    #print(max_final_area)


    rect = cv2.minAreaRect(final_contours[max_final_area_index])
    box_width = rect[1][0]
    box_height = rect[1][1]
    box_area = box_width*box_height

    LR_area = 0
    for my_area in final_area:
        if my_area/box_area >0.3:
            LR_area = LR_area + my_area
    #print(LR_area)

    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(img4,[box],-1,(0,0,255),1)

    #the distance value
    final_distance = param[0] * (LR_area*100) + param[1]
    #print(final_distance)
    #print(box)
    #bottom_left = box[0]
    #top_right = box[2]

    #print(bottom_left)
    #print(top_right)

    pl.imshow(img4)
    pl.show()

    return box[0,0], box[0,1], box[2,0], box[2,1], final_distance
#blX, blY, trX, trY, d
