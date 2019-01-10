import numpy as np
import math
import p3_utils
from scipy.special import expit

def refresh_map(Z_4, b_T_l, best_particle, w_T_b_best, MAP, odd_map, b = 4):

	B_beam_ends = np.dot(b_T_l, Z_4) #scanning data, a line of the beam ends, in body frame
	W_beam_ends = np.dot(w_T_b_best, B_beam_ends)  #transform to world frame, the line of
	# the beam ends 4*1081 matrix (x, y, z, 1)

	bool_position = W_beam_ends[2, :] >= 0.1 + 0.93  # remove the beam in the world frame that smaller than 0.93 +0.1
	W_beam_ends = W_beam_ends[:, bool_position]

	x_particle = np.ceil((best_particle[0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1 # particle is the body respect
	# to world frame, and extract the particle position (point x-y), in map coordinate, (0--1000, 0--1000)
	y_particle = np.ceil((best_particle[1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1

	for i in range(W_beam_ends.shape[1]):
		ex = np.ceil((W_beam_ends[0,i] - MAP['xmin'])/MAP['res']).astype(np.int16)-1 #in map coordinate to the pixel
		ey = np.ceil((W_beam_ends[1,i] - MAP['ymin'])/MAP['res']).astype(np.int16)-1 #in map coordinate to the pixel

		scan_section = p3_utils.bresenham2D(x_particle, y_particle, ex, ey) #a beam section for just one end
		scan_section = scan_section.astype(int)

		odd_map[scan_section[0][-1], scan_section[1][-1]] = \
			odd_map[scan_section[0][-1], scan_section[1][-1]] + math.log(b)
		# for occupied cell, wall, the end point

		odd_map[scan_section[0][1:-1], scan_section[1][1:-1]] = \
			odd_map[scan_section[0][1:-1], scan_section[1][1:-1]] + math.log(1 / b)


	odd_map[scan_section[0][0], scan_section[1][0]] = \
		odd_map[scan_section[0][0], scan_section[1][0]] + math.log(1 / b)

	P_occupied = 1- expit(-odd_map)
	bool_occupied_cells = P_occupied > 0.95  #if the probability larger than 0.8, assign it as occupied with 1
	bool_free_cells = P_occupied < 0.05 #if the probability smaller than 0.2, assign it as free with 0
	mt = bool_free_cells *(-1) + bool_occupied_cells * 1

	return mt, odd_map # which is the map contain the assigned value, 0, 1, 0.5


