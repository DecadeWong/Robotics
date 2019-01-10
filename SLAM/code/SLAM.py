import load_data

import numpy as np
from transforms3d.euler import euler2mat, mat2euler
import math
import mapping
import localization
import matplotlib.pyplot as plt
from scipy.special import expit
from texture import texture_mapping
from PIL import Image

#for test set
lidar = load_data.get_lidar('lidar/test_lidar')

#for training set
#lidar = load_data.get_lidar('lidar/train_lidar3')
lidar_pose = [x['pose'] for x in lidar]  # absolute position lidar respective to world
lidar_scan = [x['scan'] for x in lidar]  # world object respect to lidar
lidar_ts = [x['t'] for x in lidar]

#for test set
joint = load_data.get_joint('joint/test_joint')

#for training set
#joint = load_data.get_joint('joint/train_joint3')
joint_headangles = joint['head_angles']
joint_neck = joint_headangles[0]  # around z axis
joint_head = joint_headangles[1]  # around y axis
joint_ts = joint['ts']


###############################load RGB and depth data for training set0 ###########
# RGB = load_data.get_rgb('cam/RGB_0')
# RGB_It = [x['image'] for x in RGB] #RGB information
# RGB_head_angles = [[x['head_angles'].squeeze()[0], x['head_angles'].squeeze()[1]]for x in RGB]
# RGB_head_angles = np.array(RGB_head_angles)
# RGB_neck = RGB_head_angles[:, 0]
# RGB_head = RGB_head_angles[:, 1]
#
# depth = load_data.get_depth('cam/DEPTH_0')
# depth_dt = [x['depth'] for x in depth] #depth information
# texture_ts = [x['t'] for x in depth]

########################## load RGB and depth data for training set 3 ############
# RGB31 = load_data.get_rgb('cam/RGB_3_1')
# RGB32 = load_data.get_rgb('cam/RGB_3_2')
# RGB33 = load_data.get_rgb('cam/RGB_3_3')
# RGB34 = load_data.get_rgb('cam/RGB_3_4')
# RGB = RGB31 + RGB32 + RGB33 + RGB34
#
# RGB_It = [x['image'] for x in RGB] #RGB information
# RGB_head_angles = [[x['head_angles'].squeeze()[0], x['head_angles'].squeeze()[1]]for x in RGB]
# RGB_head_angles = np.array(RGB_head_angles)
# RGB_neck = RGB_head_angles[:, 0]
# RGB_head = RGB_head_angles[:, 1]
#
# depth = load_data.get_depth('cam/DEPTH_3')
# depth_dt = [x['depth'] for x in depth] #depth information
# texture_ts = [x['t'] for x in depth]

########################### load RGB and depth data for test set#############
# RGB1 = load_data.get_rgb('cam/RGB_1')
# RGB2 = load_data.get_rgb('cam/RGB_2')
# RGB3 = load_data.get_rgb('cam/RGB_3')
# RGB4 = load_data.get_rgb('cam/RGB_4')
# RGB5 = load_data.get_rgb('cam/RGB_5')
# RGB6 = load_data.get_rgb('cam/RGB_6')
# RGB7 = load_data.get_rgb('cam/RGB_7')
# RGB8 = load_data.get_rgb('cam/RGB_8')
# RGB9 = load_data.get_rgb('cam/RGB_9')
# RGB = RGB1+ RGB2 + RGB3 + RGB4 + RGB5 +RGB6 + RGB7 + RGB8 + RGB9
#
# RGB_It = [x['image'] for x in RGB] #RGB information
# RGB_head_angles = [[x['head_angles'].squeeze()[0], x['head_angles'].squeeze()[1]]for x in RGB]
# RGB_head_angles = np.array(RGB_head_angles)
# RGB_neck = RGB_head_angles[:, 0]
# RGB_head = RGB_head_angles[:, 1]
#
# depth = load_data.get_depth('cam/DEPTH')
# depth_dt = [x['depth'] for x in depth] #depth information
# texture_ts = [x['t'] for x in depth]
############################################################################

#operating the time
joint_ts = joint_ts.squeeze()
idxs = []
for lts in lidar_ts:
	checking_value = lts.squeeze()
	idx = np.where(abs(joint_ts - checking_value) == min(abs(joint_ts - checking_value)))
	idxs.append(idx[0][0])
num = len(idxs)
print(num)
#get the index of  joint associated with lidar

# in physical unit
MAP = {}
MAP['res'] = 0.1  # meters
MAP['xmin'] = -30  # meters
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']))  # DATA TYPE: char or int8

x_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
y_range = np.arange(-0.4, 0.4 + 0.1, 0.1)
#which correspongding to 1000*1000 pixel map
x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # physical x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # physical y-positions of each pixel of the map

# initialize the particles
N = 100
X = np.zeros((1,N)) #arange(MAP['xmin'], MAP['xmax'], 0.4)
Y = np.zeros((1,N)) #arange(MAP['ymin'], MAP['ymax'], 0.4)
theta = np.zeros((1,N)) #np.arange(-180, 177, 3.6) / 180 * math.pi
Xt_t = np.array([X, Y, theta]).reshape(3, N)  # where N should be 100, particles, Xt0
weight = (np.array([1/N] * N).reshape(1, N))

# initialize the map
mt = MAP['map']
odd_map = np.zeros((MAP['sizex'],MAP['sizey']))
tmt = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
# initialize the odometray
O_t = np.array([0, 0, 0]).reshape(3, 1)
trajectoryx = []
trajectoryy = []
#angles = np.arange(-135/180 * math.pi, 135/180*math.pi, 0.00436332)  # 1081 angles
ppp = np.arange(0, num, 100)#12040  #num
for k in ppp:
	angles = np.arange(-135 / 180 * math.pi, 135 / 180 * math.pi, 0.00436332)
	ranges = np.double(lidar_scan[k].squeeze())  # test first time step ranges####################### scan

	indValid = np.logical_and((ranges < 30), (ranges > 0.1)) #remove some non-qualified beams
	ranges = ranges[indValid]
	angles_2 = angles[indValid]

	xs0 = np.array([ranges * np.cos(angles_2)])  # convert polar coordinate to cartesian coordinate
	ys0 = np.array([ranges * np.sin(angles_2)])
	Z_3 = np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0)
	Z_4 = np.concatenate([Z_3, np.ones(xs0.shape)], axis=0)  # measurement 4* 1081

	Tz = np.zeros((4, 4))
	Ty = np.zeros((4, 4))
	#use the idxs to track the
	Rz = euler2mat(0, 0, joint_neck[idxs[k]], axes='sxyz')########################  neck
	Tz [0:3, 0:3] = Rz
	Tz [:, 3] = [0, 0, 0.15, 1]
	Ry = euler2mat(0, joint_head[idxs[k]], 0, axes='sxyz')  ######################    head
	Ty [0:3, 0:3] = Ry #np.dot(np.array(Rz), np.array(Ry))  # transform from lidar to body frame
	Ty [:, 3] = [0, 0, 0.33, 1]  # 0.48 is the height above the center mass, head above center mass 33cm + lidar above
	b_T_l = np.dot(Ty, Tz)

	lidar_pose_t = lidar_pose[k].squeeze()  ##################################
	# pose, odometrary, respect to world
	w_T_l = np.zeros((4, 4))  # reset the w_O_l matrix to 4*4 zero matrix
	Rw_T_l = euler2mat(0, 0, lidar_pose_t[2], axes='sxyz')  # rotation part of the wOl which is 3*3
	w_T_l[0:3, 0:3] = Rw_T_l
	w_T_l[:, 3] = [lidar_pose_t[0], lidar_pose_t[1], 1.41, 1]  # 4*4 matrix for W_O_L
	MO_t1 = np.dot(w_T_l, np.linalg.inv(b_T_l))  # matrix of Ot+1 4*4 wTb
	theta = mat2euler(MO_t1[0:3, 0:3])  # rotation matrix, get ang of x, y, z

	# above is to calculate the Ot+1
	O_t1 = np.array([MO_t1[0, 3], MO_t1[1, 3], theta[2]]).reshape(3, 1)  # O_t+1 is 3*1 vector with (x, y, theta),

	# particle filter, mapping
	Xt_t, weight, w_T_b_best, best_particle = localization.PF\
		(N, O_t, O_t1, Xt_t, weight, b_T_l, Z_4, mt, MAP, x_range, y_range, x_im, y_im)

	trajectoryx.append(np.ceil((-1*best_particle[0]- MAP['xmin'])/MAP['res']).astype(np.int16)-1)
	trajectoryy.append(np.ceil((best_particle[1]- MAP['ymin'])/MAP['res']).astype(np.int16)-1)

###################################
	# if (lidar_ts[k].squeeze() <= texture_ts[-1].squeeze()) and \
	# 		(lidar_ts[k].squeeze() >= texture_ts[0].squeeze()):
	# 	nearest_ts = min(texture_ts, key=lambda x: abs(x - lidar_ts[k]))
	# 	indx = texture_ts.index(nearest_ts)
	#
	# 	tmt = texture_mapping(w_T_b_best, RGB_neck[indx], RGB_head[indx],
	# 	                      MAP, RGB_It[indx], depth_dt[indx], tmt)
#########################
	mt, odd_map = mapping.refresh_map(Z_4, b_T_l, best_particle, w_T_b_best, MAP, odd_map)
	O_t = np.copy(O_t1)

P_occupied = 1- 1/(1+np.exp(odd_map)) #the occupied probability for each cell 1000*1000
# #or manipulate the mt which is being modified
bool_occupied_cells = P_occupied > 0.95 #if the probability larger than 0.8, assign it as occupied with 1
bool_free_cells = P_occupied < 0.05 #if the probability smaller than 0.2, assign it as free with 0
ggg = bool_free_cells *(-1) + bool_occupied_cells*1


plt.figure(1)
plt.imshow(ggg, cmap='gray_r')
plt.show()

plt.figure(2)
plt.plot(trajectoryy, trajectoryx)
plt.show()

plt.figure(3)
r1, g1, b1 = 0, 0, 0 # Original value
r2, g2, b2 = 255, 255, 255 # Value that we want to replace it with
red, green, blue = tmt[:,:,0], tmt[:,:,1], tmt[:,:,2]
mask = (red == r1) & (green == g1) & (blue == b1)
tmt[:,:,:3][mask] = [r2, g2, b2]
img_t = Image.fromarray(tmt, 'RGB')
plt.imshow(img_t)
plt.show()