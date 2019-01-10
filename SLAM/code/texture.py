import numpy as np
from transforms3d.euler import euler2mat, mat2euler
import load_data


IRCalib = load_data.getIRCalib()
Matrix_IRCalib = np.array([[IRCalib['fc'][0], IRCalib['ac']*IRCalib['fc'][0], IRCalib['cc'][0]],
                           [0, IRCalib['fc'][1], IRCalib['cc'][1]],
                           [0,0,1]])
RGBCalib = load_data.getRGBCalib()
Matrix_RGBCalib = np.array([[RGBCalib['fc'][0], RGBCalib['ac']*RGBCalib['fc'][0], RGBCalib['cc'][0]],
                           [0, RGBCalib['fc'][1], RGBCalib['cc'][1]],
                           [0,0,1]])
exIR_RGB = load_data.getExtrinsics_IR_RGB()
#transform from IR frame to RGB frame
RGB_T_IR = np.zeros((4,4))
RGB_T_IR[0:3, 0:3] = exIR_RGB['rgb_R_ir']
RGB_T_IR[:, 3] = [exIR_RGB['rgb_T_ir'][0], exIR_RGB['rgb_T_ir'][1],
                         exIR_RGB['rgb_T_ir'][2], 1]

def sensor2body(neck, head):
	Tz = np.zeros((4, 4))
	Ty = np.zeros((4, 4))
	#use the idxs to track the
	Rz = euler2mat(0, 0, neck, axes='sxyz')########################  neck
	Tz [0:3, 0:3] = Rz
	Tz [:, 3] = [0, 0, 0.07, 1]
	Ry = euler2mat(0, head, 0, axes='sxyz')  ######################    head
	Ty [0:3, 0:3] = Ry #np.dot(np.array(Rz), np.array(Ry))  # transform from lidar to body frame
	Ty [:, 3] = [0, 0, 0.33, 1]  # 0.48 is the height above the center mass, head above center mass 33cm + lidar above
	b_T_c = np.dot(Ty, Tz)
	return b_T_c

def texture_mapping(w_T_b_best,RGB_neck, RGB_head, MAP, RGB_It, depth_dt, tmt):
	b_T_rgb = sensor2body(RGB_neck, RGB_head)
	# b_T_ir = sensor2body(IR_neck[i], IR_head[i])
	#
	w_T_c_rgb = np.dot(w_T_b_best, b_T_rgb) #where w_T_b is from the particle filter

	# w_T_c_ir = np.dot(w_T_b_best, b_T_ir)
	o_T_c = np.array([[0, -1, 0, 0],
	                  [0, 0, -1, 0],
	                  [1, 0, 0, 0],
	                  [0, 0, 0, 1]])

	pixel_IR_array = np.array(list(np.ndindex((IRCalib['nxy'][1], IRCalib['nxy'][0]))))
	#create a column vector, the index, (424*512)*2, in pixel frame
	added_one = np.ones((1, IRCalib['nxy'][1]* IRCalib['nxy'][0]))
	#print(np.concatenate([pixel_IR_array.T, added_one], axis=0))
	pixel_IR_array3 = np.concatenate([pixel_IR_array.T, added_one], axis=0)
	u = np.copy(pixel_IR_array3[1, :])
	v = np.copy(pixel_IR_array3[0, :])
	pixel_IR_array3[0, :] = u
	pixel_IR_array3[1, :] = v
	#still in pixel frame, (u, v, 1)

	normal_optical_depth = np.dot(np.linalg.inv(Matrix_IRCalib), pixel_IR_array3)
	#(Xo/Zo, Yo/Zo, 1) 3* (424*512), in focal plane
	row_depth_dt = depth_dt.reshape(1, IRCalib['nxy'][1]* IRCalib['nxy'][0]).squeeze()
	#matrix 424 * 512 to row vector 1* (424*512)     #important have depth value
	IRoptical_depth = np.concatenate([(normal_optical_depth * row_depth_dt), added_one], axis =0)
	#(Xo, Yo, Zo, 1) in IR optical frame 4 * (424*512)
	RGBoptical_depth = np.dot(RGB_T_IR, IRoptical_depth) ########
	row_depth_dt = RGBoptical_depth[2, :]


	pixel_RGB_array_depth = np.dot(Matrix_RGBCalib, RGBoptical_depth[0:3, :]/row_depth_dt)
	#(u, v, 1) 3* (424*512)
	x = np.round(pixel_RGB_array_depth)
	indvalid = ~np.isnan(x)
	pixel_RGB_array_depth = x[indvalid].reshape(3, -1).astype(int)

	# print(np.max(pixel_RGB_array_depth[1,:])) #v
	# print(np.max(pixel_RGB_array_depth[0,:])) #u
	row_depth_dt = row_depth_dt[indvalid[0, :]]

	shrink_indx = np.logical_and(pixel_RGB_array_depth[1, :]>=0, pixel_RGB_array_depth[1, :]<=1079)
	pixel_RGB_array_depth = pixel_RGB_array_depth [:, shrink_indx]
	#(u, v, 1)
	row_depth_dt = row_depth_dt[shrink_indx]

	shrink_indx2 = np.logical_and(pixel_RGB_array_depth[0, :]>=0, pixel_RGB_array_depth[0, :]<=1919)
	pixel_RGB_array_depth = pixel_RGB_array_depth[:, shrink_indx2]
	#(u, v, 1)
	row_depth_dt = row_depth_dt[shrink_indx2]

	column_u = pixel_RGB_array_depth[0, :]
	column_v = pixel_RGB_array_depth[1, :]

	Extract_RGB_It = RGB_It[column_v, column_u, :]
	Extract_RGB_It = Extract_RGB_It.T
	#depth: row_depth_dt, pixel_RGB_array_depth, RGB_It are aligned
	#now the point in RGB_pixel has RGB values and associated depth value
	#it can now be transform to world frame
	#trans_back to focal plane

	normal_RGBoptical_depth = np.dot(np.linalg.inv(Matrix_RGBCalib), pixel_RGB_array_depth)
	#(Xo/Zo, Yo/Zo, 1)
	RGBoptical_depth = normal_RGBoptical_depth * row_depth_dt
	#(Xo, Yo, Zo)
	# print(RGBoptical_depth.shape[1])
	RGBoptical_depth = np.concatenate([RGBoptical_depth, np.ones((1, RGBoptical_depth.shape[1]))], axis = 0)
	#(Xo, Yo, Zo, 1)
	RGBcamera_depth = np.dot(np.linalg.inv(o_T_c), RGBoptical_depth)/1000
	#(Xc, Yc, Zc, 1) to meter
	world_depth = np.dot(w_T_c_rgb, RGBcamera_depth)
	#(Xw, Yw, Zw, 1)

	idx = world_depth[2] <  0.01
	#update color, depth, stick to ground
	world_depth = world_depth[:, idx] #(Xw, Yw, Zw, 1)
	world_RGB = Extract_RGB_It[:, idx]
	map_x = np.ceil((world_depth[0, :]-MAP['xmin'])/MAP['res']).astype(np.int16)-1
	map_y = np.ceil((world_depth[1, :]-MAP['ymin'])/MAP['res']).astype(np.int16)-1

	final_idx_x = map_x < MAP['sizex']
	map_x = map_x[final_idx_x]
	map_y = map_y[final_idx_x]
	world_RGB = world_RGB[:, final_idx_x]

	final_idx_y = map_y < MAP['sizey']
	map_x = map_x[final_idx_y]
	map_y = map_y[final_idx_y]
	world_RGB = world_RGB[:, final_idx_y]

	my_x = map_x.tolist()
	my_y = map_y.tolist()

	tmt[my_x, my_y, 0] = world_RGB[0, :].tolist()
	tmt[my_x, my_y, 1] = world_RGB[1, :].tolist()
	tmt[my_x, my_y, 2] = world_RGB[2, :].tolist()
	return tmt




# def dead_reckon(O_t, O_t1, best_particle_t):
# 	delta = O_t1 - O_t
# 	best_particle_t1 = best_particle_t + delta
# 	return best_particle_t1
#load data
#['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','image']
# RGB = load_data.get_rgb('cam/RGB_0')
# RGB_It = [x['image'] for x in RGB] #RGB information
# RGB_head_angles = [[x['head_angles'].squeeze()[0], x['head_angles'].squeeze()[1]]for x in RGB]
# RGB_head_angles = np.array(RGB_head_angles)
# RGB_neck = RGB_head_angles[:, 0]
# RGB_head = RGB_head_angles[:, 1]
# #print(RGB_It[0].shape)
# #['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','depth']
# depth = load_data.get_depth('cam/DEPTH_0')
# depth_dt = [x['depth'] for x in depth] #depth information
# IR_head_angles = [[x['head_angles'].squeeze()[0], x['head_angles'].squeeze()[1]]for x in depth]
# IR_head_angles = np.array(IR_head_angles)
# IR_neck = IR_head_angles[:, 0]
# IR_head = IR_head_angles[:, 1]
# texture_ts = [x['t'] for x in depth]