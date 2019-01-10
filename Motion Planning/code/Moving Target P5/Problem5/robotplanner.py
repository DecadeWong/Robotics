import numpy as np
import math
import time

def robotplanner(envmap, robotpos, targetpos):
  numofdirs = 8
  dX = [-1, -1, -1, 0, 0, 1, 1, 1]
  dY = [-1,  0,  1, -1, 1, -1, 0, 1]
  
  # failed to find an acceptable move
  newrobotpos = np.copy(robotpos)
  
  # for now greedily move towards the target, 
  # but this is the gateway function for your planner 
  mindisttotarget = 1000000
  for dd in range(numofdirs):
    newx = robotpos[0] + dX[dd]
    newy = robotpos[1] + dY[dd]
  
    if (newx >= 0 and newx < envmap.shape[0] and newy >= 0 and newy < envmap.shape[1]):
      if(envmap[newx, newy] == 0):
        disttotarget = math.sqrt((newx-targetpos[0])**2 + (newy-targetpos[1])**2)
        if(disttotarget < mindisttotarget):
          mindisttotarget = disttotarget
          newrobotpos[0] = newx
          newrobotpos[1] = newy
  return newrobotpos

