import numpy as np
from transforms3d.euler import euler2mat, mat2euler
import p3_utils
from numpy import unravel_index

def mysoftmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum() # only difference


def localization_preditction (N, O_t, O_t1, Xt_t):
	mu = 0
	delta = O_t1 - O_t #delta with noise
	# Xt1_t = Xt_t + np.tile(delta, N) + np.tile(delta, N) + 0.1*np.random.randn(3,N)
	Xt1_t = Xt_t + np.tile(delta, N)+np.array([np.random.normal(mu, 0.095, N),
	                                           np.random.normal(mu, 0.095, N),
	                                           np.random.normal(mu, 0.095, N)])  # number of N particles, 3*N
	#with output weight_t+1\t = weight_t\t
	return Xt1_t


def localization_update(weight, mt, N, Z_4, b_T_l, Xt1_t, MAP, x_range, y_range, x_im,y_im):
	# should not be included in update step, but in upper level loop
	#which correspongding to 1000*1000 pixel map
	test_line = np.dot(b_T_l, Z_4) # 4*1081, the position of line it is a solid line
	corrs = []# 1* N, each particle corresponding a correlations
	for j in range(N):
		p = Xt1_t[:, j]
		#p represent the particle, state
		w_T_b= np.zeros((4, 4))
		Rz_particle = euler2mat(0, 0, p[2], axes='sxyz')
		w_T_b[0:3, 0:3] = Rz_particle
		w_T_b[:, 3] = [p[0], p[1], 0.93, 1]  # 4*4 matrix

		Y = np.dot(w_T_b, test_line) # the measurement converted to one particle state 4*1081
		bool_position = Y[2, :] >= 0.1
		Y = Y[:, bool_position]

		corr_matrix = p3_utils.mapCorrelation(mt, x_im,y_im, Y[0:3, :], x_range, y_range) # a matrix data
		corr = np.max(corr_matrix)

		location = unravel_index(corr_matrix.argmax(), corr_matrix.shape)
		delta_x = (location[0] - 4)*MAP['res'] # in physical delta
		delta_y = (location[1] - 4)*MAP['res']
		# Xt1_t[0, j] = Xt1_t[0, j] - delta_x
		# Xt1_t[1, j] = Xt1_t[1, j] - delta_y
		Xt1_t[0, j] = p[0] + delta_x
		Xt1_t[1, j] = p[1] + delta_y
		corrs.append(corr)

	ph = mysoftmax(np.array(corrs)).reshape(1,N)
	#ph(zt|x,m) have to be transformed to a vector, vector
	weight_t1 = weight * ph / np.sum(weight * ph) #.reshape(1, N) # a vector
	return weight_t1, Xt1_t


def resampling(Xt1_t, weight_t1, N):
	new_X = np.zeros((3, N))
	new_weight = np.tile(1/N, N).reshape(1, N)
	j = 0
	c = weight_t1[0, 0] # weight_t1 is normalized
	for k in range(N):
		u = np.random.uniform(0, 1/N) #uniform distribution
		beta = u + k/N #scan each part in the circle
		while beta > c :
			j = j +1
			c = c + weight_t1[0, j] # increasing the decision section length
		new_X[:, k] = Xt1_t[:, j] #if Beta is smaller than many times, put this repeated particles j in new set
		# k=1, k=2, k=3, may all have same particles X[1]
	return new_weight, new_X


def PF(N, O_t, O_t1, Xt_t, weight, b_T_l, Z_4, mt, MAP, x_range, y_range, x_im,y_im):
	Xt1_t = localization_preditction(N, O_t, O_t1, Xt_t)
	weight_t1, Xt1_t1= localization_update(weight, mt, N, Z_4, b_T_l, Xt1_t, MAP, x_range, y_range, x_im, y_im)

	Max_weight_index = np.argmax(weight_t1)
	Max_weight_particle = Xt1_t1[:, Max_weight_index] #3*1

	w_T_b_decided = np.zeros((4, 4))
	Rz_particle_decided = euler2mat(0, 0, Max_weight_particle[2], axes='sxyz')
	w_T_b_decided[0:3, 0:3] = Rz_particle_decided
	w_T_b_decided[:, 3] = [Max_weight_particle[0], Max_weight_particle[1], 0.93, 1]  # 4*4 matrix for W_O_L

	Neff = 1/np.dot(weight_t1.reshape(1,N), weight_t1.reshape(N,1))
	if Neff < 5:
		weight_t1, Xt1_t1 = resampling(Xt1_t1, weight_t1, N)
	return Xt1_t1, weight_t1, w_T_b_decided, Max_weight_particle