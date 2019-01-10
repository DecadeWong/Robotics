from my_function import mul, inv, log, exp, norm
import math
import numpy as np
from transforms3d import euler

def avgqtn(set_q, weight, sigma=0.0001):
    '''import set quaternion and set weight, error sigma, starting point
    #q1 = euler.euler2quat(0, 0, 170/180*math.pi)
    #q2 = euler.euler2quat(0, 0, -101/180*math.pi)
    #q3 = euler.euler2quat(0, 0, 270/180*math.pi)

    #set_q = [q1, q2, q3]
    #weight = [1/3, 1/3, 1/3]

    #a = avgqtn(set_q, weight)
    #k = euler.quat2euler(a)
    '''
    qt_bar = set_q[0]

    #weight = [0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    n = len(weight) #size of weight

    while True:
        ev = np.zeros((n, 3)) # n is the size of the set of Q n*3
        for q, w, i in zip(set_q, weight, range(len(weight))):
            qi_e = mul(inv(qt_bar), q)
            qie = np.array(qi_e)/norm(qi_e)
            e = np.array(2)* log(qie.tolist()) #  4 dimension
            ev_i = e[1:4] # 3 dimension
            if norm(ev_i) != 0:
                ev_fi = (-math.pi + math.fmod(norm(ev_i) + math.pi, math.pi * 2)) / \
                        norm(ev_i) * np.array(ev_i) * w
                ev[i, :] = ev_fi

        E = np.sum(ev, axis=0) #final ev after summation 1*n
        qt1_bar = mul(qt_bar, exp([0, E[0]/2, E[1]/2, E[2]/2]))
        qt_bar = qt1_bar #list
        if norm(E) < sigma:
            break
    return qt_bar




#q1 = euler.euler2quat(0, 170/180*math.pi, 0)
#q2 = euler.euler2quat(0, -101/180*math.pi, 0)
#q3 = euler.euler2quat(0,  270/180*math.pi, 0)

#set_q = [q1, q2, q3]
#weight = [1/3, 1/3, 1/3]

#a = avgqtn(set_q, weight)
#k = euler.quat2euler(a)

#print(k)






