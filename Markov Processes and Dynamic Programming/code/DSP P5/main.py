
# coding: utf-8

# In[54]:


import sys, os
import numpy as np
import itertools
import copy


def look_up(path, edge_table):
    path_cost = 0
    #for node_current, node_next in zip(path[0::1], path[1::1]):
    for current in range(len(path)-1):
        node_current = path[current]
        node_next = path[current+1]
        mask= np.logical_and(edge_table[:, 0]==node_current, 
            edge_table[:, 1]==node_next)
        path_cost = path_cost + edge_table[mask, 2]
    return path_cost  


def inherit (paths, costs, father_nodes, edge_table):
    """useing father_nodes, find its correspongding children_nodes for 
    each father node in father nodes"""
    paths_r2c = [] #path from root to children
    fathers_new = []
    #change the children_nodes as father nodes for next iter 
    costs_r2c = []#cost from root to children, including children
    for father_node in father_nodes:
        mask = edge_table[:, 0] == father_node
        child_nodes = edge_table[mask, 1].tolist()
        f2c_costs = edge_table[mask, 2].tolist() 
        #one father to its children, a step cost
        for child_node in child_nodes:
            costs_r2c.append(costs[father_nodes.index(father_node)]
                            +f2c_costs[child_nodes.index(child_node)])
            fathers_new.append(child_node)
            temp = copy.deepcopy(paths[father_nodes.index(father_node)])
            temp.append(child_node)
            paths_r2c.append(temp)
    return paths_r2c, costs_r2c, fathers_new


def filter_node (paths_r2c, costs_r2c, fathers_new):
    """shrink the same children nodes, keep the children nodes unique
    which now is to operate father_nodes_new"""
    costs_updated = []
    paths_updated = []
    nodes_set = list(set(fathers_new))# now children nodes is the father nodes updated
    for node in nodes_set:
        mask = (np.array(fathers_new) == node).tolist()
        fil_costs = list(itertools.compress(costs_r2c, mask))
        fil_paths = list(itertools.compress(paths_r2c, mask))
        min_cost = min(fil_costs)
        idx = fil_costs.index(min_cost)
        costs_updated.append(min_cost)
        paths_updated.append(fil_paths[idx])
    return paths_updated, costs_updated, nodes_set#which is father_updated


def filter_path (paths_updated, costs_updated, fathers_updated, paths, edge_table):
    """trace back previous routine, get its cost"""
    fathers_updated2 = []
    costs_updated2 = []
    paths_updated2 = []
    for node in fathers_updated:
        cost_now = costs_updated[fathers_updated.index(node)]
        costs_old = []
        for path in paths:
            if node in path:
                idx = path.index(node)+1
                costs_old.append(look_up(path[:idx],edge_table))
        if costs_old == []:
            fathers_updated2.append(node)
            costs_updated2.append(costs_updated[fathers_updated.index(node)])
            paths_updated2.append(paths_updated[fathers_updated.index(node)])
            continue
        elif min(costs_old) >= cost_now:
            fathers_updated2.append(node)
            costs_updated2.append(costs_updated[fathers_updated.index(node)])
            paths_updated2.append(paths_updated[fathers_updated.index(node)])
    return paths_updated2, costs_updated2, fathers_updated2


def find_path(paths, costs, father_nodes, edge_table, terminal, my_path, my_cost):
    paths_r2c, costs_r2c, fathers_new = inherit (paths, costs, father_nodes, edge_table)
    
    paths_updated, costs_updated, fathers_updated = filter_node (paths_r2c, costs_r2c, fathers_new)
    
    paths_updated2, costs_updated2, fathers_updated2 = filter_path (paths_updated, costs_updated, fathers_updated, paths, edge_table)

    if terminal in fathers_updated2:
        terminal_idx = fathers_updated2.index(terminal)
        my_path.append(paths_updated2[terminal_idx])
        my_cost.append(costs_updated2[terminal_idx]) 
        del fathers_updated2[terminal_idx]
        del costs_updated2 [terminal_idx]
        del paths_updated2 [terminal_idx]   
        if min(my_cost) <= min (costs_updated2):
            return my_path, my_cost
        else:
            return find_path(paths_updated2, costs_updated2, fathers_updated2, edge_table, terminal, my_path, my_cost)
    else:
        return find_path(paths_updated2, costs_updated2, fathers_updated2, edge_table, terminal, my_path, my_cost)

     
with open('./data/input1.txt', 'r') as file_object:
    contents = file_object.read().splitlines()
    
results1 = []
for line in contents:
    line = line.split()
    line = list(map(float, line))
    results1.append(line)    
    
with open('./data/input2.txt', 'r') as file_object:
    contents = file_object.read().splitlines()
    
results2 = []
for line in contents:
    line = line.split()
    line = list(map(float, line))
    results2.append(line)     

resultss = [results1, results2]    

f = open('output_dsp.txt', 'w')
for results in resultss:
    paths = [results[1]]
    costs = [0]
    father_nodes = copy.deepcopy(results[1])
    edge_table = np.array(results[3:])
    terminal = results[2][0]
    my_path = []
    my_cost=[]
    mp,mc=find_path(paths, costs, father_nodes, edge_table, terminal, my_path, my_cost)
    final_path = mp[0]
 
    step_cost = []
    for i in range(len(final_path)):
        step_cost.append(look_up(final_path[:i+1], edge_table))
        
    final_step_cost =list(map(int, step_cost))
    
    print(final_step_cost)
    print(final_path)
    final_step_cost_str = str(final_step_cost)
    final_path_str = str(final_path)
    
    f.write(final_path_str)
    f.write('\n')
    f.write(final_step_cost_str)
    f.write('\n')
f.close()
    







