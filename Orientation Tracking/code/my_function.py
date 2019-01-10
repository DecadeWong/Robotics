import numpy as np
import math
from dipy.core.geometry import cart2sphere, sphere2cart

def mul(q, p):
    '''multiplication of two quaternion'''
    qs = np.array(q[0]); ps = np.array(p[0])
    qv = np.array(q[1:4]); pv = np.array(p[1:4])
    s = np.array(qs*ps) - np.dot(qv,pv)
    v = np.array(qs*pv )+ np.array(ps*qv) + np.cross(qv, pv)
    answer = [s, v[0], v[1], v[2]]
    return answer


def log(q):
    '''log of a quaternion'''
    qs = np.array(q[0]); qv = np.array(q[1:4])
    norm_q = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    norm_qv = math.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if norm_qv !=0:
        s = math.log(norm_q)
        vs = (math.acos(qs/norm_q)/norm_qv)
        v = vs * np.array(qv)
        answer = [s, v[0], v[1], v[2]]
    else:
        answer = [0, 0, 0, 0]
    return answer


def exp(q):
    '''exponential of quaternion'''
    qs = np.array(q[0])
    qv = np.array(q[1:4])
    norm_qv = math.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    s = math.cos(norm_qv)*math.exp(qs)
    if norm_qv !=0:
        vs = math.sin(norm_qv)/norm_qv *math.exp(qs)
        v = vs * np.array(qv)
        answer = [s, v[0], v[1], v[2]]
    else:
        answer=[1, 0, 0, 0]
    return answer


def cjgt(q):
    '''conjugate of quaternion'''
    qv = np.array(q[1:4])
    s = q[0]
    v = -qv
    answer = [s, v[0], v[1], v[2]]
    return answer


def inv (q):
    '''inverse of quaternion'''
    #ab_q = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    #q_bar = np.array(cjgt(q))/ ab_q
    q_bar = np.array(cjgt(q))/ np.dot(q, q)
    return q_bar.tolist()


def rv(q, s):
    '''rotation of quaternion'''
    q_bar = np.array(inv (q))
    r1 = mul (q, s)
    r_final = mul(r1,q_bar)
    return r_final

def norm(q):
    ''' Return norm of quaternion'''
    return math.sqrt(np.dot(q, q))

def d3tod4 (r):
    q = exp([0, r[0]/2, r[1]/2, r[2]/2 ])

    return q

def d4tod3 (q):
    r=2*log(q)
    return r[1:4]



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







