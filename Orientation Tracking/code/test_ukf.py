import pickle
import sys
import time
import math
import matplotlib.pyplot as plt
from my_avgqtn import avgqtn
import numpy as np

import my_function as qtn
from transforms3d.quaternions import mat2quat
from transforms3d.euler import mat2euler, quat2euler
import PM


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

dataset="13"
#cfile = "cam/cam" + dataset + ".p"
ifile = "imu/imuRaw" + dataset + ".p"
#vfile = "vicon/viconRot" + dataset + ".p"

ts = tic()
#camd = read_data(cfile)
imud = read_data(ifile)
#vicd = read_data(vfile)
toc(ts,"Data import")

imu_values = imud['vals']
imu_ts = imud['ts']
#vic_rots = vicd['rots']
#vic_ts = vicd['ts']

Vref = 3300; sensitivity = 3.33
biasW = imu_values[3:6, 0:10].sum(axis=1)/10
scale_factor = Vref / 1023 / sensitivity * (math.pi/180)

imu_orienstate = []
vic_orienstate = []
estimate_orienstate = []
Rq_imu_step = []
#num = min(imu_values.shape[1], vic_rots.shape[2])

###################step for prediction
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
  #ypr = mat2euler(vic_rots[:, :, i])  # xyz transform the rotation matrix into euler
  #vic_orienstate.append(ypr)  # get series of the euler state of vicomn


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

  euler_ut_t = quat2euler(ut_t)
  estimate_orienstate.append(euler_ut_t)



plt.figure(1)
p1 = plt.subplot(311)
#p1.plot(np.array(vic_orienstate)[:, 0])
p1.plot(np.array(imu_orienstate)[:, 0])
p1.plot(np.array(estimate_orienstate)[:, 0])
plt.title('Wx')
plt.legend(['imu', 'est'])

p2 = plt.subplot(312)
#p2.plot(np.array(vic_orienstate)[:, 1])
p2.plot(np.array(imu_orienstate)[:, 1])
p2.plot(np.array(estimate_orienstate)[:, 1])
plt.title('Wy')
plt.legend(['imu', 'est'])


p3 = plt.subplot(313)
#p3.plot(np.array(vic_orienstate)[:, 2])
p3.plot(np.array(imu_orienstate)[:, 2])
p3.plot(np.array(estimate_orienstate)[:, 2])
plt.title('Wz')
plt.legend(['imu', 'est'])
plt.show()