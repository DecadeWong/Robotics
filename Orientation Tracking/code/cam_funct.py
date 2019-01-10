import pickle
import sys
import time
import math
import matplotlib.pyplot as plt
from my_avgqtn import avgqtn
import numpy as np

import my_function as qtn
from dipy.core.geometry import cart2sphere, sphere2cart
from transforms3d.euler import mat2euler, quat2euler
import PM
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

imu_values = imud['vals']
imu_ts = imud['ts']
vic_rots = vicd['rots']
vic_ts = vicd['ts']
graph_value = camd['cam']
graph_ts = camd['ts']



def get_mask(vert, hori):

	delta_v = 45 / 240  # vertical leads to row
	delta_h = 60 / 320
	ds = math.pi / 180  # degree scale
	mask = np.zeros((vert, hori, 3))
	for i in range(vert):  # first scanning row
		for k in range(hori):  # scanning column
			# above convert one pixel into sphere coordinate
			phi = (67.5 + delta_v / 2 + delta_v * i) * ds  # relate to row, which is the latitude, phi, 0 to pi
			lmd = (-30 + delta_h / 2 + delta_h * k) * ds  # relate to column, which is the longitude, xy,lmd
			v = sphere2cart(1, phi, lmd)
			mask[i, k, :] = [v[0].tolist(), v[1].tolist(), v[2].tolist()]
	return mask  # in cartesian

def insert_graph (d2mask, graph_i, vic_rots, Panorama_Field):
	cts_shifted2d = np.dot(vic_rots, d2mask)  # make orientation this is achieved in 2d, 3*(240*320)
	# cts_shifted3d = d2mask.reshape(np.roll(d3mask.shape,1)).transpose(1,2,0) #transform back from 2d to 3d
	r, phi_s, lmd_s = cart2sphere(cts_shifted2d[0, :], cts_shifted2d[1, :], cts_shifted2d[2, :])  # final location in
	# sphereical coordinate at 2d
	new_row = phi_s / math.pi * 4*240
	new_column = (lmd_s + math.pi) / (2 * math.pi) * 6*320
	#print(new_row)
	new_location2d = np.array([new_row, new_column])
	new_location3d= new_location2d.reshape(np.roll((240, 320, 2), 1)).transpose(1, 2, 0)  # transform back from 2d to 3d

	for i in range(new_location3d.shape[0]):  # first scanning row 240
		for k in range(new_location3d.shape[1]):  # scanning column
			row = int(new_location3d[i, k, 0])
			column = int(new_location3d[i, k, 1])
			Panorama_Field[row, column, :] = graph_i[i, k, :]
	return Panorama_Field


def find_nearest(array, values):
	indices = np.abs(np.subtract.outer(array, values)).argmin(0)
	return indices


idx = find_nearest(vic_ts[0], graph_ts)
vert, hori, RGB, camsize = graph_value.shape
Panorama_Field = np.zeros((4*240, 6*320, 3), dtype=np.uint8)
d3mask = get_mask(vert, hori)  # constant mask can be used in multi times, with cartesian value
d2mask = d3mask.transpose(2, 0, 1).reshape(3, -1)  # transform from 3d to 2d


for j, k in zip(range(camsize), idx[0, :]):
#for j in [0, 50, 84, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 850, 900, 950, 1680]:
	graph = graph_value[:, :, :, j] #scanning the graph from one cam doc
	Panorama_Field = insert_graph (d2mask, graph, vic_rots[:, :, k], Panorama_Field)
	#vic_rot is the orientation of that input graph


img = Image.fromarray(Panorama_Field, 'RGB')
plt.imshow(img)
plt.show()