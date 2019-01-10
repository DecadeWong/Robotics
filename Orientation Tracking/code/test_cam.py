import pickle
import sys
import time
import math
import matplotlib.pyplot as plt
import numpy as np

import my_function as qtn
from transforms3d.euler import quat2euler, quat2mat
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

dataset="10"
cfile = "cam/cam" + dataset + ".p"
ifile = "imu/imuRaw" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)

toc(ts,"Data import")

imu_values = imud['vals']
imu_ts = imud['ts']
cam_values = camd['cam']
cam_ts = camd['ts']

Vref = 3300; sensitivity = 3.33
biasW = imu_values[3:6, 0:10].sum(axis=1)/10
scale_factor = Vref / 1023 / sensitivity * (math.pi/180)

imu_orienstate = []
estimate_orienstate = []
Rq_imu_step = []

###################step for prediction, update for estimation
qt = [1, 0, 0, 0]
ut_t = np.array([1, 0, 0, 0]) # mean of first quaternion, first time first qt
covt_t = 0.0001 * np.eye(3) # covariance of first quaternion, first time
Q = 0.0001 * np.eye(3) #noise of motion and observation
scale_factorg = Vref / 1023 / 300
biasacc = imu_values[0:3, 0:10].sum(axis=1)/10 - np.array([0, 0, 1])/scale_factorg


for i in range(imu_values.shape[1]-1):
  accx_imu = -1 * (imu_values[0, i+1] - biasacc[0]) * scale_factorg
  accy_imu = -1 * (imu_values[1, i+1] - biasacc[1]) * scale_factorg
  accz_imu = (imu_values[2, i+1] - biasacc[2]) * scale_factorg
  acc_imu = [accx_imu, accy_imu, accz_imu] #each imu step for acc

  Wz_imu = 0.5 * (imu_values[3, i] - biasW[0]) * scale_factor * (imu_ts[0, i+1] - imu_ts[0, i])
  Wx_imu = 0.5 * (imu_values[4, i] - biasW[1]) * scale_factor * (imu_ts[0, i+1] - imu_ts[0, i])
  Wy_imu = 0.5 * (imu_values[5, i] - biasW[2]) * scale_factor * (imu_ts[0, i+1] - imu_ts[0, i])
  W_imu = [Wx_imu, Wy_imu, Wz_imu]
  Rq_imu = [0] + W_imu  # rotaion in quaternion


  qt_1 = qtn.mul(qt, qtn.exp(Rq_imu))  # orientation at state t_1
  qt = qt_1
  euler_qt_1 = quat2euler(qt_1)  # xyz
  imu_orienstate.append(euler_qt_1)  # transform the quaternion orientation in to euler states
  E = PM.sigma(covt_t, Q)
  Pqt_1_bar, Pcov, Pqt1, e = PM.prediction(ut_t, E, Rq_imu) #where e is the error vector
  #qt+1 bar is the predicted mean, cov is the predicted mean, Pqt1 is the predicted 7 sigma points
  #measurement
  zt1_bar, cov_vv, zt1 = PM.measurement(Pqt1)
  #E_qt1 = PM.sigma(Pcov, 0)
  #kalem gain
  kt1 = PM.kgain(e, zt1_bar, zt1, cov_vv)
  #update
  qt1_updated, covt1_updated = PM.update(kt1, acc_imu, zt1_bar, cov_vv, Pqt_1_bar, Pcov)
  covt_t = covt1_updated
  ut_t = qt1_updated
  # euler_ut_t = quat2euler(ut_t)
  # estimate_orienstate.append(euler_ut_t)
  rot_matrix = quat2mat(ut_t)
  estimate_orienstate.append(rot_matrix)

print(len(estimate_orienstate))
print(cam_values.shape)

idx = qtn.find_nearest(imu_ts[0], cam_ts)
print(idx.shape)

vert, hori, RGB, camsize = cam_values.shape
print(camsize)
Panorama_Field = np.zeros((4*240, 6*320, 3), dtype=np.uint8)
d3mask = qtn.get_mask(vert, hori)  # constant mask can be used in multi times, with cartesian value
d2mask = d3mask.transpose(2, 0, 1).reshape(3, -1)  # transform from 3d to 2d



for j, k in zip(range(911), idx[0, 0:911]):
#for j in [0, 50, 84, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 850, 900, 950, 1680]:
	graph = cam_values[:, :, :, j] #scanning the graph from one cam doc
	print(k)
	if k >=len(estimate_orienstate):

		Panorama_Field = qtn.insert_graph (d2mask, graph, estimate_orienstate[k-1], Panorama_Field)
	else:
		Panorama_Field = qtn.insert_graph(d2mask, graph, estimate_orienstate[k], Panorama_Field)
	#vic_rot is the orientation of that input graph


img = Image.fromarray(Panorama_Field, 'RGB')
plt.imshow(img)
plt.show()