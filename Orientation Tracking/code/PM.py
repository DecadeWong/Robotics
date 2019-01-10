import numpy as np
import my_function as qtn
from my_avgqtn import avgqtn

def sigma(covt_t, Q):
	'''get sigma point, and each point is in 3d
	the sigma point is based on the gaussian distribution,
	in this condition seven sigma points can describe the whole gaussian
	distribution. as the quaternion orientation is fuzzy, is gaussian like
	'''
	Wnoise = np.linalg.cholesky(3 * (covt_t + Q))  # 3x3 matrix
	#Emean = qtn.d4tod3(mean)
	#the sigma point is the value of qt + the noise, 7 points
	E0 = np.array([0, 0, 0]) #E0 is the main orientation sigma point, left are 6 other sigma points
	E1 = E0 + Wnoise[:, 0]
	E4 = E0 - Wnoise[:, 0]
	E2 = E0 + Wnoise[:, 1]
	E5 = E0 - Wnoise[:, 1]
	E3 = E0 + Wnoise[:, 2]
	E6 = E0 - Wnoise[:, 2]
	E = [E0, E1, E2, E3, E4, E5, E6]
	return E #list of arrays each array is 3d


def prediction(qt, E, step):
	'''where Rq_imu_step is the quaternion rotation for each step
	based on t predict t+1
	this function is based on sigma point, noise added
	'''
	#motion model
	qt1 = []
	e = [] #error vector for qt+1   7
	for Ei in E: #predict each sigma point from time t to t+1
		qt1_i = qtn.mul(qt, qtn.mul(qtn.exp
			([0, 0.5 * Ei[0], 0.5 * Ei[1], 0.5 * Ei[2]]), qtn.exp(step)))
		qt1.append(qt1_i) #qt1 means at qt+1 state there is 7 sigma points
		#in qt+1 the first element is the sigma point transformed from estimated qt mean

	#calculate the mean and covariance
	weight = [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
	qt_1_bar = avgqtn(qt1, weight) #mean value of the predicted sigma point, a list value
	cov = np.zeros((3, 3)) #covariance is 3*3 matrix since quaternion is shrinked from 4 to 3
	for qt1_i, i in zip(qt1, range(len(qt1))):
		#q = qtn.mul(qt1_i, qtn.inv(qt_1_bar))
		q = qtn.mul(qtn.inv(qt_1_bar), qt1_i)
		ei = qtn.d4tod3(q)  # ei is a 3 dimension list
		eiv = np.array(ei).reshape((3,1))
		e.append(eiv)
		if i == 0:
			cov = 2 * np.dot(eiv, eiv.T)
		else:
			cov = cov + 1/6 * np.dot(eiv, eiv.T) #covariance value of the predicted sigma point
	return qt_1_bar, cov, qt1, e # where the qt1bar is the mean of the state qt+1 predicted
	#qt1 contain seven sigma point


def measurement (qt1):
	'''with predicted qt+1, and imu_acc at this qt+1 step'''
	#measurement
	zt1 = [] #which is my measurement
	#noise_t = np.array(imu_acc_step) - np.array(zt_bar) # zt - zt_bar
	for qt1_i in qt1: # seven element
		qzt1_i = qtn.mul(qtn.mul(qtn.cjgt(qt1_i), [0, 0, 0, 1]), qt1_i)
		#world to body frame qt-1 *v * qt, if it is body to world qt * v * qt-1
		zt1_i = np.array(qzt1_i[1:4])  #+ noise_t #noise is 3d
		zt1.append(zt1_i.tolist())
	zt1_bar = 1/6 * np.sum(zt1[1:7], axis = 0) + 0 * np.array(zt1[0])
	covzz = np.zeros((3,3))
	for zt1_i, i in zip(zt1, range(len(zt1))):
		z = np.array(zt1_i) - zt1_bar
		zt1_iv = np.array(z).reshape((3, 1))
		if i == 0:
			covzz = 2 * np.dot(zt1_iv, zt1_iv.T)
		else:
			covzz = covzz + 1/6 * np.dot(zt1_iv, zt1_iv.T) #covariance
	cov_vv = covzz + 0.0001*np.eye(3)
	return zt1_bar, cov_vv, zt1


def kgain (e, zt1_bar, zt1, cov_vv ):
	'''calculate cov_xz, it is the covariance between predicted sigma point, and error of measurement'''
	cov_xz = np.zeros((3,3))
	for i in range(len(e)):
		et1 = np.array(zt1[i] - zt1_bar)
		if i ==0:
			cov_xz = 2 * np.dot(np.reshape(e[i], (3,1)), np.reshape(et1, (1,3)))
			#calculate Pxz
		else:
			cov_xz = cov_xz + 1 / 6 * np.dot(np.reshape(e[i], (3,1)), np.reshape(et1, (1,3)))
	kt1 = np.dot(cov_xz, np.linalg.inv(cov_vv))
	return kt1  # 3*3 matrix


def update(kt1, imu_acc_t1, zt1_bar, cov_vv, Pqt1_bar, cov_predicted):
	'''kt1 is a 3*3 matrix'''
	inov_t1 = np.array (imu_acc_t1 - zt1_bar)
	k_inovt1 = np.dot(kt1, inov_t1)

	qt1_updated = qtn.mul(Pqt1_bar,
		qtn.exp([0, 0.5*k_inovt1[0], 0.5*k_inovt1[1], 0.5*k_inovt1[2]]))
	k_Pvv = np.dot(kt1 , cov_vv)
	covt1_updated = cov_predicted - np.dot(k_Pvv, kt1.T)
	return qt1_updated, covt1_updated













