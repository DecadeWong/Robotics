import numpy as np
import math

def targetplanner(envmap, robotpos, targetpos, basepos, movetime):
  dX = [-1, 0, 0, 1]
  dY = [ 0, -1, 1, 0]
  
  # failed to find an acceptable move
  newtargetpos = np.copy(targetpos)
  
  for mind in range(movetime):
    # generate a move at random in 4 directions
    maxdist = 0
    for iter in range(2):
      dd = np.random.randint(0,4)
      newx = targetpos[0] + dX[dd]
      newy = targetpos[1] + dY[dd]
      
      if (newx >= 0 and newx < envmap.shape[0] and newy >= 0 and newy < envmap.shape[1]):
        dist = math.sqrt((newx-basepos[0])**2 + (newy-basepos[1])**2)
        if( (envmap[newx, newy] == 0) and (dist > maxdist) ):
          newtargetpos[0] = newx
          newtargetpos[1] = newy
          maxdist = dist
  return newtargetpos

