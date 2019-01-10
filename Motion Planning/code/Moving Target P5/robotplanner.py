import numpy as np
import numpy as np
import math
import time
from collections import defaultdict
class RobotPlanner:
    def __init__(self, envmap, start, max_lookahead=1000):
        #the max lookahead can be set to large for complex map
        self.envmap = envmap
        self.g_dist = np.ones(envmap.shape) * np.inf
        self.h_dist = np.zeros(envmap.shape) #0
        self.g_dist[start[0], start[1]] = 0
        self.numofdirs = 8
        self.dX = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.dY = [-1,  0,  1, -1, 1, -1, 0, 1]
        self.max_lookahead = max_lookahead
        # to find neighbor and also to backtrack
        self.finalpath= [(start[0], start[1])]

    def initialization(self, currpos):
        self.CLOSE = set()
        self.OPEN = {(currpos[0], currpos[1])}
        self.g_dist = np.ones(self.envmap.shape) * np.inf
        self.g_dist[currpos[0], currpos[1]] = 0


    def find_i_open(self, targetpos, factor = 1):
        """find the i I want to expand"""
        min_fi = None
        mycoord = None
        for coord in self.OPEN:
            h_i = None
            if self.h_dist[coord[0], coord[1]] == 0:
                dx_sq = (targetpos[0] - coord[0])**2
                dy_sq = (targetpos[1] - coord[1])**2
                h_i = math.sqrt(dx_sq+dy_sq)
            else:
                h_i = self.h_dist[coord[0], coord[1]]
            f_i = factor * h_i + self.g_dist[coord[0], coord[1]]
            if min_fi == None and mycoord == None:
                min_fi = f_i
                mycoord = coord
            elif min_fi >= f_i:
                min_fi = f_i
                mycoord = coord
        return mycoord, min_fi

    def trace_path(self, parents, goal, start):
        path = [goal]
        child = goal
        while start not in path:
            path.append(parents[child])
            child = parents[child]
        return path[::-1] 

    def find_neighbors(self, myi):
        neighbors = []
        f2cs = []
        for dd in range (self.numofdirs):
            newx = myi[0] + self.dX[dd]
            newy = myi[1] + self.dY[dd]
            if (newx >= 0 and newx < self.envmap.shape[0] \
                and newy >= 0 and newy < self.envmap.shape[1]):
                    if(self.envmap[newx, newy] == 0):
                        f2c = round(math.sqrt(self.dX[dd]**2 + self.dY[dd]**2),2)
                        neighbors.append((newx, newy))
                        f2cs.append(f2c)
        return neighbors, f2cs

    def RTAA_star(self,  targetpos, currpos, factor =1 ):
        self.initialization(currpos)
        count = 0
        parents = {}
        # local A * algorithm
        while True: 
            myi, f_i = self.find_i_open(targetpos)# find minimum i of fi in open set
            if count == self.max_lookahead:
                break
            self.OPEN.remove(myi)
            self.CLOSE.add(myi)
            if (targetpos[0], targetpos[1]) in self.CLOSE:
                break
            # iterate through all neighbors
            children, f2cs = self.find_neighbors(myi)
            for child, f2c in zip(children, f2cs):
                if child not in self.CLOSE:
                    start2child_cost = self.g_dist[myi[0], myi[1]]+ f2c
                    if self.g_dist[child[0], child[1]]> start2child_cost:
                        self.g_dist[child[0], child[1]] = start2child_cost
                        #update the g value
                        self.OPEN.add(child)
                        parents[child] = myi 
            count += 1
        # update h_dist values h[i] = f[s_bar] - g[i]
        node_j, f_j = self.find_i_open(targetpos)
        
        for i in self.CLOSE:
        	self.h_dist[i[0], i[1]] = f_j - self.g_dist[i[0], i[1]]

        mypath = self.trace_path(parents, node_j, (currpos[0], currpos[1]))
        newrobotpos = mypath[1]
        #print(newrobotpos)
        self.finalpath.append(newrobotpos)
        return newrobotpos


      

 





