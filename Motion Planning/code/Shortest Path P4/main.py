
# coding: utf-8

# In[2]:


#os.path.abspath("something.exe")
#os.path.dirname() 
#os.path.dirname(os.path.abspath(__file__))
import os
import copy
import math
from collections import defaultdict, namedtuple

class Graph():
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distance = {}
    def add_node(self, value,):
        self.nodes.add(value)
    def add_edge(self, father_node, child_node, distance):
        self.edges[father_node].append(child_node)
        self.distance[(father_node, child_node)] = distance
        
class shortest_path():
    def __init__(self, edges_table, coords_table):
        self.start = edges_table[1] # a list return
        self.goal = edges_table[2]
        self.coords = coords_table
    def Dijkstra(self, graph):
        start = self.start[0]
        goal = self.goal[0]
        open_set = copy.deepcopy(graph.nodes)
        parents = {}
        visted_nodes = {start: 0} #store the point I visted, and the cost from start to key value
        iter_num = 0
        while goal in open_set:
            lst_val = []
            iter_num = iter_num + 1
            lst_nodesval = list(visted_nodes.values())
            lst_nodeskey = list(visted_nodes.keys())#1000
            filter_keys = [x for x in lst_nodeskey if x in open_set]
            for key in filter_keys:
                lst_val.append(visted_nodes[key])#999
            min_cost = min(lst_val)
            idx = lst_val.index(min_cost)
            min_key = filter_keys[idx]
            open_set.remove(min_key)
            children = graph.edges [min_key]
            for child in children:
                start2child_cost = min_cost + graph.distance[(min_key, child)]    
                if child not in visted_nodes: #insert a new key of child and its value to 
                    visted_nodes[child] =  start2child_cost
                    parents[child] = min_key
                else:
                    if start2child_cost < visted_nodes[child]:
                        visted_nodes[child] =  start2child_cost
                        parents[child] = min_key
        cost = visted_nodes[goal] 
        #path = self.trace_path(parents, goal, start)
        return cost, iter_num #path
    def A_star(self, graph, factor):
        start = self.start[0]
        goal = self.goal[0]
        open_set = {start}
        close_set = set()
        visted_nodes = {start: 0} #store the point I visted, and the cost from start to key 
        parents = {}
        goal_coord = self.coords[int(goal - 1)]
        iter_num = 0
        while goal not in close_set:
            min_fi = None
            min_i = None
            iter_num = iter_num + 1
            for i in open_set: #strating from 1
                i_coord = self.coords[int(i -1)] #parent location
                x = (goal_coord[0] - i_coord[0])**2
                y = (goal_coord[1] - i_coord[1])**2
                h_i = math.sqrt(x+y)#this is the heuristic distance
                f_i = factor * h_i + visted_nodes[i] # hi + gi
                if min_fi == None and min_i == None:
                    min_fi = f_i
                    min_i = i
                elif min_fi >= f_i:
                    min_fi = f_i
                    min_i = i # this is node we want to expand
            min_cost = visted_nodes[min_i] 
            open_set.remove(min_i)
            close_set.add (min_i)
            children = graph.edges [min_i]
            for child in children: 
                if child not in close_set:
                    start2child_cost = min_cost + graph.distance[(min_i, child)]    
                    if child not in visted_nodes: #insert a new key of child and its value to 
                        visted_nodes[child] =  start2child_cost
                        parents[child] = min_i
                        open_set.add(child)
                    else:
                        if start2child_cost < visted_nodes[child]:
                            visted_nodes[child] =  start2child_cost
                            parents[child] = min_i
                            open_set.add(child)
        cost = visted_nodes[goal] 
        #path = self.trace_path(parents, goal, start)
        return cost, iter_num #path
    def trace_path(self, parents, goal, start):
        path = [goal]
        child = goal
        while start not in path:
            path.append(parents[child])
            child = parents[child]
        return path[::-1]

if __name__ == '__main__':
    path = os.getcwd() + '/src'
    files = os.listdir(path + '/input')
    tables = defaultdict(list)
    for file in files:
        with open (path + '/input/' + file, 'r') as file_obj:
            contents = file_obj.read().splitlines()
            #without splitlines, the content is a whole string, class str
            #with splitlines, a list of string, create a list
            for line in contents: #operating a list
                line = line.split() #list
                #split the line seperated by ','
                line = list(map(float, line))
                tables[file].append(line)            
    #the information get here is tables            
    Data1 = namedtuple('Data1', ['input', 'coords']) 
    Data2 = namedtuple('Data2', ['input', 'coords'])
    Data3 = namedtuple('Data3', ['input', 'coords'])
    data1 = Data1(tables['input_1.txt'], tables['coords_1.txt'])
    data2 = Data2(tables['input_2.txt'], tables['coords_2.txt'])
    data3 = Data3(tables['input_3.txt'], tables['coords_3.txt'])
    Data = [data1, data2, data3]

    if os.path.exists('src/output_costs.txt') and os.path.exists('src/output_numiters.txt'):
        os.remove('src/output_costs.txt')
        os.remove('src/output_numiters.txt')
    j = 0
    for data in Data:
        j = j + 1
        edges_table = data.input
        coords_table = data.coords
        #graph created
        graph = Graph()
        for line in edges_table[3:]:
            father = line[0]
            child = line[1]
            distance = line[2]
            graph.add_node(father)
            graph.add_node(child)
            graph.add_edge(father, child, distance) 
        short_path = shortest_path(edges_table, coords_table)
        cost_D, iternum_D  = short_path.Dijkstra(graph)
        cost_A1, iternum_A1 = short_path.A_star(graph, 1)
        cost_A2, iternum_A2 = short_path.A_star(graph, 2)
        cost_A3, iternum_A3 = short_path.A_star(graph, 3)
        cost_A4, iternum_A4 = short_path.A_star(graph, 4)
        cost_A5, iternum_A5 = short_path.A_star(graph, 5)

        print(cost_D, cost_A1, cost_A2,cost_A3, cost_A4, cost_A5)
        print(iternum_D, iternum_A1, iternum_A2, iternum_A3, iternum_A4, iternum_A5)
        with open('src/output_costs.txt', 'a') as file_costs:
            #file_costs.write('input' + str(j) + '\n')
            ccccc= str(cost_D) + '      ' + str(cost_A1) + '      ' + str(cost_A2) + '      ' + str(cost_A3) + '      ' + str(cost_A4) + '      ' + str(cost_A5)
            file_costs.write(ccccc + '\n')
            
#             file_costs.write(str(cost_D) + '      ' + str(cost_A1) + '      ' + \
#                              str(cost_A2) + '      ' + str(cost_A3) + '      ' + \
#                              str(cost_A4) + '      ' + str(cost_A5) + '\n')
            #file_costs.write(str([cost_D, cost_A1, cost_A2,cost_A3, cost_A4, cost_A5]) + '\n')
        with open('src/output_numiters.txt', 'a') as file_numiters:
            #file_numiters.write('input' + str(j) + '\n')
            iiiii= str(iternum_D) + '      ' + str(iternum_A1) + '      ' + str(iternum_A2) + '      ' + str(iternum_A3) + '      ' + str(iternum_A4) + '      ' + str(iternum_A5)
            file_numiters.write(iiiii + '\n')
            
            
            
            #file_numiters.write(str([iternum_D, iternum_A1, iternum_A2, iternum_A3, iternum_A4, iternum_A5]) + '\n')
#             file_costs.write(str(iternum_D) + '      ' + str(iternum_A1) + '      ' + \
#                              str(iternum_A2) + '      ' + str(iternum_A3) + '      ' + \
#                              str(iternum_A4) + '      ' + str(iternum_A1) + '\n')

