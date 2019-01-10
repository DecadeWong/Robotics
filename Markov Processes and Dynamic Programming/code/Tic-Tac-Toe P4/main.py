
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import matplotlib.pyplot as plt
from itertools import permutations, compress
import random

def find_empty(state_ls):
    mask = (np.array(state_ls) == '1').tolist()
    pro_typ = [0,1,2,3,4,5,6,7,8]
    idx_ls = list(compress(pro_typ, mask))
    #print(idx_ls)
    idx = random.choice(idx_ls)
    return idx

def simulation_player_first (table, itert):
    count_loss =0
    count_win = 0
    for i in range(itert):
        state_child_str =  '1'*9 #initialize
        state_child_ls = list(state_str)
        rdm_idx = random.randint(0, 8)
        state_child_ls[rdm_idx] = '2' #now initialized
        for step in [1, 2, 3, 4, 5, 6, 7, 8]:
            #print(state_child_ls)
            if step in [1,3,5, 7]: #[2,4,6,8]
                idx = find_empty(state_child_ls)
                state_child_ls[idx] = '0'
                reward = find_reward(state_child_ls)
                if reward == -1:
                    count_loss = count_loss + 1
                    break  
            elif step in [2,4,6,8]: #[1,3,5,9] #[2,4,6,8]
                table = linkers[step]
                state_child_str = ''.join(state_child_ls)
                state_father_str = table[state_child_str]
                reward = find_reward(state_father_str)
                if reward == 1:
                    count_win = count_win + 1
                    break
                else:
                    state_child_ls = list(state_father_str)   
    return count_loss, count_win
   
def simulation_machine_first (table, itert):
    count_loss =0
    count_win = 0
    for i in range(itert):
        state_child_str =  '1'*9 #initialize
        state_child_ls = list(state_str)
        rdm_idx = random.randint(0, 8)
        state_child_ls[rdm_idx] = '0' #now initialized
        for step in [1, 2, 3, 4, 5, 6, 7, 8]:
            #print(state_child_ls)
            if step in [2,4,6,8]:
                idx = find_empty(state_child_ls)
                state_child_ls[idx] = '0'
                reward = find_reward(state_child_ls)
                if reward == -1:
                    count_loss = count_loss + 1
                    break  
            elif step in [1,3,5,7]: #[1,3,5,9] #[2,4,6,8]
                table = linkers[step]
                state_child_str = ''.join(state_child_ls)
                state_father_str = table[state_child_str]
                reward = find_reward(state_father_str)
                if reward == 1:
                    count_win = count_win + 1
                    break
                else:
                    state_child_ls = list(state_father_str)   
    return count_loss, count_win

def stage_buidler(playermove_step, machinemove_step):
    x = '2'
    o = '0'
    empty = '1'
    chess_patterns = x* playermove_step + o*machinemove_step +     empty*(9 -playermove_step-machinemove_step  )
    return {''.join(chess) for chess in permutations(chess_patterns)}

def check_win (chess_on_board, who):
    if who == 'player':
        total_value = 6
    elif who == 'machine':
        total_value = 0
    checker = 0
    for i in range(chess_on_board.shape[0]):
        if chess_on_board[i, :].sum() == total_value:
            checker =1
            break
        elif chess_on_board[:, i].sum() ==total_value:
            checker =1
            break
    if chess_on_board.trace() == total_value or    np.fliplr(chess_on_board).trace() == total_value:
        checker = 1
    return checker

def find_reward(chess_pattern):
    chess_pattern = list(map(int, chess_pattern))
    chess_on_board = np.array(chess_pattern).reshape(3,3)
    machine_win = check_win(chess_on_board, 'machine')
    player_win = check_win(chess_on_board, 'player')
    if player_win and not machine_win:
        return 1
    elif machine_win and not player_win:
        return -1
    return 0

def last_reward(chess_patterns, consider_losing):
    reward = {}
    for chess_pattern in chess_patterns:
        checker = find_reward(chess_pattern)
        chess_pattern
        if checker ==1:#player win
            reward[chess_pattern] = 1
        elif checker ==-1: #machine win
            if consider_losing ==1:
                reward[chess_pattern] = -1
            elif consider_losing ==0: #not consider losing
                reward[chess_pattern] = 0
        elif checker ==0: #tie
            reward[chess_pattern] = 0
    return reward

def inherit (father_states_dict, child_states_dict, reward, stage, who_first):
    child_rewards = {}
    linker = {}
    if who_first == 'player':
        for child in child_states_dict:
            c2f_link = []
            aaaa=[]
            for father in father_states_dict:
                cmpr = np.array(list(child)) == np.array(list(father))
                if sum(cmpr) == 8:      # find child's=stage8, father=stage9 childern's father
                    c2f_link.append(reward[father])
                    aaaa.append(father)
            if stage in [8,6,4,2]: #player move  
                child_rewards[child] = max(c2f_link)
                idx = c2f_link.index(max(c2f_link))
                father_string = aaaa[idx]
                linker[child] = father_string
            elif stage in [7,5,3,1]: #machine move
                if find_reward(child) ==1:
                    child_rewards[child] = 1
                else:
                    child_rewards[child] = np.mean(c2f_link)
                
    elif who_first == 'machine':
        for child in child_states_dict:
            c2f_link = []
            aaaa=[]
            for father in father_states_dict:
                cmpr = np.array(list(child)) == np.array(list(father))
                if sum(cmpr) == 8:      # find child's=stage8, father=stage9 childern's father
                    c2f_link.append(reward[father])
                    aaaa.append(father)
            if stage in [7,5,3,1]: #player move  
                child_rewards[child] = max(c2f_link)
                idx = c2f_link.index(max(c2f_link))
                father_string = aaaa[idx]
                linker[child] = father_string
            elif stage in [8,6,4,2,0]: #machine move
                if find_reward(child) ==1:
                    child_rewards[child] = 1
                else:
                    child_rewards[child] = np.mean(c2f_link)
    return child_rewards, linker


#who_first = 'machine'
#consider_losing = 0 #when it is 1, the losing is considered, 0 is not consider losing
#not consider losing, and that is reward at last is win 1 and other is 0

f = open('output_ttt.txt', 'w')
itert =10000
for who_first in ['player', 'machine']:
    for consider_losing in [0, 1]:
        if who_first == 'player':
            linkers = {}
            father_states_dict = stage_buidler(playermove_step=5, machinemove_step=4)
            reward = last_reward(father_states_dict, consider_losing)
            for stage in range(8, 0, -1):
                step = stage//2
                child_states_dict = stage_buidler(playermove_step=stage-step, machinemove_step=step)
                reward, linker = inherit (father_states_dict, child_states_dict, reward, stage, 'player')
                father_states_dict = child_states_dict
                linkers[stage] = linker 
            count_loss, count_win = simulation_player_first (linkers, itert )

        elif who_first == 'machine':
            linkers = {}
            father_states_dict = stage_buidler(playermove_step=4, machinemove_step=5)
            reward = last_reward(father_states_dict, consider_losing)
            for stage in range(8, -1, -1):   
                step = stage//2
                child_states_dict = stage_buidler(playermove_step=step, machinemove_step=stage-step)
                reward, linker = inherit (father_states_dict, child_states_dict, reward, stage, 'machine')
                father_states_dict = child_states_dict
                linkers[stage] = linker
            count_loss, count_win = simulation_machine_first (linkers, itert )
        
        
        #print(who_first)
        #print(consider_losing)
        print(max(reward.values()))
        print(count_loss/itert)
        print(count_win/itert)
        f.write(str(max(reward.values())))
        f.write('\n')
        f.write(str(count_win/itert - count_loss/itert))
        f.write('\n')
f.close() 
      
    

