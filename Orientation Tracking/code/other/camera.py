import pickle
import sys
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from dipy.core.geometry import sphere2cart, cart2sphere
from PIL import Image


def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
      if sys.version_info[0] < 3:
        d = pickle.load(f)
      else:
        d = pickle.load(f, encoding='latin1')  # need for python 3
    return d

dataset="1"
cfile = "cam/cam" + dataset + ".p"
ifile = "imu/imuRaw" + dataset + ".p"
vfile = "vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

vic_rots = vicd['rots']
vic_ts = vicd['ts']
graph_value = camd['cam']
graph_ts = camd['ts']

# def insert_graph (graph_i, vert, hori, vic_rots, Panorama_Field):
#     '''insert one rotated graph in panorama'''
#     #scanning pixel for just one graph
#     delta_v = 45 / 240  # vertical leads to row
#     delta_h = 60 / 320
#     ds = math.pi / 180  # degree scale
#     for i in range(vert): #first scanning row
#         for k in range(hori): #scanning column
#             # above convert one pixel into sphere coordinate
#             phi = (67.5 + delta_v / 2 + delta_v * i) * ds #relate to row, which is the latitude, phi, 0 to pi
#             lmd = (-30 + delta_h / 2 + delta_h * k) * ds  #relate to column, which is the longitude, xy,lmd
#             #convert the pixel of sphere coordinate into cartesian
#             v = sphere2cart(1, phi, lmd)
#             cts_shifed = np.dot(vic_rots, v) #shifted picture in cartesian, do shifting
#             r, phi_s, lmd_s = cart2sphere(cts_shifed[0], cts_shifed[1], cts_shifed[2])#shifed in sphere coordinate
#             new_row = int(phi_s/math.pi*2*240) #location in panorama field
#             new_column = int((lmd_s+math.pi)/(2*math.pi)*3*320)
#             #now, within created new blanket of same graphic size, insert the new location
#             Panorama_Field[new_row, new_column, :] = graph_i[i,k, :]
#             #the pixel in original graph at i,k, is now at new_row, and new_column, with corresponding RGB
#             ############################################################################################
#             ###above method just insert one graph in panorama field, need to iteration to add more graph
#     return Panorama_Field

def get_mask(vert, hori):
    delta_v = 45 / 240  # vertical leads to row
    delta_h = 60 / 320
    ds = math.pi / 180  # degree scale
    mask = np.zeros((vert, hori, 3))
    for i in range(vert): #first scanning row
        for k in range(hori): #scanning column
            # above convert one pixel into sphere coordinate
            phi = (67.5 + delta_v / 2 + delta_v * i) * ds #relate to row, which is the latitude, phi, 0 to pi
            lmd = (-30 + delta_h / 2 + delta_h * k) * ds #relate to column, which is the longitude, xy,lmd
            v = sphere2cart(1, phi, lmd)
            mask[i,k,:] = [v[0].tolist(), v[1].tolist(), v[2].tolist()]
    return mask #in cartesian

vert, hori, RGB, camsize = graph_value.shape
Panorama_Field = np.zeros((180, 360, 3))

d3mask = get_mask(vert, hori) #constant mask can be used in multi times, with cartesian value
d2mask = d3mask.transpose(2,0,1).reshape(3, -1) #transform from 3d to 2d


cts_shifted2d = np.dot(vic_rots[:, :, 0], d2mask) #make orientation this is achieved in 2d, 3*(240*320)
#cts_shifted3d = d2mask.reshape(np.roll(d3mask.shape,1)).transpose(1,2,0) #transform back from 2d to 3d

r, phi_s, lmd_s = cart2sphere(cts_shifted2d[0,:], cts_shifted2d[1,:], cts_shifted2d[2,:])#final location in
# sphereical coordinate at 2d
new_row = int(phi_s/math.pi*180)
new_column = int((lmd_s+math.pi)/(2*math.pi)*360)
new_location = np.array(new_row, new_column)
final_mask = new_location.reshape(np.roll((vert, hori, 2),1)).transpose(1,2,0) #transform back from 2d to 3d


for i in range(final_mask.shape[0]): #first scanning row 240
    for k in range(final_mask.shape[1]): #scanning column
        row = final_mask[i, k, 0]
        column = final_mask[i,k,1]

        Panorama_Field[row, column, :]  = graph_i[i, k, :]















# ########iterate all graphs
# Panorama_Field = np.zeros((2*240, 3*320, 3), dtype=np.uint8) #height 4*240, and width 6*320, but thi is in sphere coordinate
# #where = np.where(vic_ts[0] > graph_ts[0, 0])
# #have to converted to cylinder coordinate
# vert, hori, RGB, camsize = graph_value[:, :, :, :].shape
# #size = min(camsize, vic_rots.shape[2])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# for j in range(3):
# #for j in [0, 50, 84, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 850, 900, 950, 1680]:
#     #[0, 850, 900, 950, 1500, 1550, 1600, 1620, 1640, 1660, 1680]:
#         #range(size - 1):
#         #[0, 50, 84, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 850, 900, 950, 1680]:
#     graph_j = graph_value[:, :, :, j] #scanning the graph from one cam doc
#     #img =graph_j [::10, ::10, :]
#     Panorama_Field = insert_graph (graph_j, vert, hori, vic_rots[:, :, 3*j+165], Panorama_Field)
#     #vic_rot is the orientation of that input graph
# #print(Panorama_Field)
# img = Image.fromarray(Panorama_Field, 'RGB')
# #img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# plt.imshow(img)
# plt.show()




#vic_orienstate = []
#phi is from 0 to pi, lmd is from -pi to pi on the xy plane
# num = vic_rots.shape[2]
# for i in range(num):
#
#   ypr = mat2euler(vic_rots[:, :, i]) #xyz transform the rotation matrix into euler
#   vic_orienstate.append(ypr) #get series of the euler state of vicomn
#
# #graph the data
# p1 = plt.subplot(311)  #vicon rotation
# p1.plot(vic_orienstate)
# plt.title('ground truth')
# plt.legend(['Wx', 'Wy', 'Wz'])



