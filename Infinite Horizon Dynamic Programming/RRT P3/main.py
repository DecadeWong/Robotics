from collections import defaultdict, namedtuple
import math, random, sys, pygame, os
import shapely.geometry as sg
import copy

class RRT_Star():
    def __init__(self, start, terminal, obstacles):
        self.start = start
        self.terminal = terminal
        self.obstacles = obstacles
        #below is the graph structure
        self.vertices = {start}
        #initialize the Vertices with start point
        self.edges = set()
        #initialize edge as empty
        #bellow is some untility
        self.Xrand = None
        self.g = {start : 0}
        self.find = False
        self.scalex = 20
        self.scaley = 20
        self.movex = 15
        self.movey = 15


    def __dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
    def __Nearest(self):
        """find nearest node in the vertices to Xrand"""
        min_dist = float('inf')
        min_node = None
        for vertex in self.vertices:
            mydist = self.__dist(vertex, self.Xrand)
            if mydist < min_dist:
                min_dist = mydist
                min_node = vertex
        return self.g[min_node], min_node #min_node = Xnearest
    
    def __Steer(self, Xnearest, EPSILON = 1.5):
        if self.__dist(Xnearest, self.Xrand) < EPSILON:
            Xnew = self.Xrand
        elif self.__dist(Xnearest, self.terminal) < EPSILON:
            Xnew = self.terminal
        else:
            theta = math.atan2(self.Xrand[1]-Xnearest[1]
                               ,self.Xrand[0]-Xnearest[0])
            Xnew = (Xnearest[0] + EPSILON*math.cos(theta), Xnearest[1] + EPSILON*math.sin(theta))
        cost_Xnew = self.g[Xnearest] + self.__dist(Xnew, Xnearest)
        return cost_Xnew, Xnew
    
    def __collision_checker(self, point1, point2):
        """check the extend line whether intersect with obstacle
        from point1 to point2"""
        extend_line = [point1, point2]
        extend_line = sg.LineString(extend_line)
        collision = None
        for obstacle in self.obstacles:
            my_obstacle = sg.Polygon(obstacle)
            x = my_obstacle.intersection(extend_line)
            if type(x) == sg.linestring.LineString or type(x) == sg.multilinestring.MultiLineString:
                collision = True
                return collision
            elif type(x) == sg.collection.GeometryCollection:
                collision = False
        return collision
        
    def __Near(self, Xnew, beta = 2): 
        N = len(self.vertices)
        radius = beta #* math.log(N)/N
        Xnear = []
        for vertex in self.vertices:
            mydist = self.__dist(Xnew, vertex)
            if mydist < radius:
                Xnear.append(vertex)
        return Xnear
    
    def Extend(self):
        temp_vertices = copy.deepcopy(self.vertices)
        temp_edges = copy.deepcopy(self.edges)
        cost_Xnearest, Xnearest = self.__Nearest()
        cost_Xnew, Xnew = self.__Steer(Xnearest)
        self.g[Xnew] = cost_Xnew
        #which is cost of Xnearest + epsilon
        if self.__collision_checker(Xnearest, Xnew) == False:
            #from Xnearest to Xnew
            #no collision
            temp_vertices.add(Xnew)
            xmin = copy.deepcopy(Xnearest) # need deepcopy maybe
            Xnear = self.__Near(Xnew) #return a list
            for xnear in Xnear:
                if self.__collision_checker(xnear, Xnew) == False:
                    mycost = self.g[xnear] + self.__dist(xnear, Xnew)
                    if mycost < cost_Xnew:
                        xmin = xnear
                        self.g[Xnew] = mycost   
            #finally we get the xmin and updated the gXnew
            temp_edges.add((xmin, Xnew)) #here if Xnew = terminal
            myxmin = ((xmin[0] + self.movex) * self.scalex, (-xmin[1] + self.movey) * self.scaley)
            myXnew = ((Xnew[0] + self.movex) * self.scalex, (-Xnew[1] + self.movey) * self.scaley)###
            pygame.draw.line(screen, black, myxmin, myXnew)

            if Xnew == self.terminal:
                self.find = True                
            else:
                myXnear = [x for x in Xnear if x != xmin]
                #follwoing is the rewiring step
                for xnear in myXnear:
                    if self.__collision_checker(Xnew, xnear) == False and\
                     self.g[xnear] > self.g[Xnew] + self.__dist(Xnew, xnear):
                        for edge in temp_edges:
                            if edge[1] == xnear:
                                xparent = edge[0]
                                temp_edges.remove((xparent, xnear))
                                temp_edges.add((Xnew, xnear)) # now Xnew is its xnear parents
                                self.g[xnear] = self.g[Xnew] + self.__dist(Xnew, xnear)
                                myxparent = ((xparent[0] + self.movex) * self.scalex, (-xparent[1] + self.movey) * self.scaley)
                                myxnear = ((xnear[0] + self.movex) * self.scalex, (-xnear[1] + self.movey) * self.scaley)
                                pygame.draw.line(screen, white, myxparent, myxnear)
                                pygame.draw.line(screen, black, myXnew, myxnear)

        pygame.display.update()                  
        self.vertices = copy.deepcopy(temp_vertices)
        self.edges = copy.deepcopy(temp_edges)

    def sample(self):
        self.Xrand = (random.uniform(-15,15),random.uniform(-15,15))

    @staticmethod
    def modify_obs (obstacles, scalex, scaley, movex, movey):
        myobstacles = []
        for obstacle in obstacles:
            myobstacle = []
            for node in obstacle:
                myobstacle.append(((node[0] + movex) *scalex, (-node[1] + movey) * scaley))
            myobstacles.append(myobstacle)     
        return myobstacles
        

if __name__ == "__main__":
    #read data
    tables = []
    path = os.getcwd() + '/src'
    myfiles = os.listdir(path + '/data')
    with open (path + '/data/' + myfiles[0], 'r') as file_obj:
        contents = file_obj.read().splitlines()
        #without splitlines, the content is a whole string, class str
        #with splitlines, a list of strings, create a list
        for line in contents: #operating a list
            #print(line)
            line = line.split(',') #list
            myline = []
            #split the line seperated by ','
            for coord in line:
                coord = coord.split()
                mycoord = tuple(map(float, coord))
                myline.append(mycoord)
            tables.append(myline) 
    start = tables[0][0]
    terminal = tables[1][0]
    obstacles = tables[2:]
    

    #modifiy the obstacles in window scale    
    X_dim = 600
    Y_dim = 600
    window = [X_dim, Y_dim]
    pygame.init()
    screen = pygame.display.set_mode(window)
    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (0, 0, 255)
    green = (0,255,0)
    red = (255,0,0)
    pink = (200, 20, 240)
    screen.fill(white)
    scalex = 20
    scaley = 20
    movex = 15
    movey = 15


    myobstacles = RRT_Star.modify_obs(obstacles, scalex, scaley, movex, movey)# modify my obstacle
    for myobs in myobstacles:
        pygame.draw.polygon(screen, blue, myobs)

    mystart = (int((start[0] + movex) *scalex), int((-start[1] + movey) * scaley))
    myterminal = (int((terminal[0] + movex) *scalex), int((-terminal[1] + movey) * scaley))
    pygame.draw.circle(screen, red, mystart, 8)#start point
    pygame.draw.circle(screen, green, myterminal, 8)#terminal point
    pygame.display.update()
    
    #create object, and main function
    myRRTStar = RRT_Star(start, terminal, obstacles)
    i = 0; N = 5000
    while i<10000000:
        if i % 500 == 0:
            myRRTStar.Xrand = copy.deepcopy(terminal)
        else:
            myRRTStar.sample() # create a new sample
        i =  i + 1
        myRRTStar.Extend()
        if myRRTStar.find:
            print('the terminal found')
            break       
    print('done')
    print(i)
    

    #post-processing
    mypath = [terminal]
    child = terminal
    if myRRTStar.find:
        while start not in mypath:
            for edge in myRRTStar.edges:
                if edge[1] == child:
                    father = edge[0]
                    mypath.append(father)
                    myfather = ((father[0] + movex) * scalex, (-father[1] + movey) * scaley)
                    mychild = ((child[0] + movex) * scalex, (-child[1] + movey) * scaley)
                    pygame.draw.line(screen, pink, myfather, mychild, 5)
                    child = copy.deepcopy(father)
        pygame.display.update()
        mypath = mypath[::-1]

print('finish')
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False