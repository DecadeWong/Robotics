import numpy as np
import math
from collections import namedtuple, defaultdict
from scipy.stats import multivariate_normal
import scipy
# import mdptoolbox.mdp as mdp
import copy
#initialize paramters settings
# MDPConstants = namedtuple('MDPConstants', ['delta_t','n_1', 'n_2', 'n_u', 'theta_max', 'v_max', 'u_max'])
# #theta is for X1, angular velocity is for X2, control is for u
# PendulumConstants = namedtuple('PendulumConstants', ['a', 'b', 'sigma', 'k', 'r', 'gamma'])
# #the constants that describe the pendulum
# DiscreteSpace = namedtuple('DiscreteSpace', ['X1_matrix', 'X2_matrix', 'U', 'Flat_board'])
# #The space describe the state space and control space

# myMDP = MDPConstants(delta_t=0.1, n_1=80, theta_max= math.pi, v_max=8, u_max=8, n_2=20, n_u=50)
# myPendulum = PendulumConstants(a=1, b=0, sigma=np.eye(2) * 0.1, k=1, r=0.01, gamma=0.3)

# X1 = np.linspace(-myMDP.theta_max, myMDP.theta_max, myMDP.n_1) # one D array for X1
# X2 = np.linspace(-myMDP.v_max, myMDP.v_max, myMDP.n_2) # one D array for X2

# X1_matrix, X2_matrix = np.meshgrid(X1, X2) #create the state space for X meshgrid format
# U = np.linspace(-myMDP.u_max, myMDP.u_max, myMDP.n_u)[:, None] # one D array for control U
# # where row number is for X2, column number is for X1
# Flat_board = [i for i in zip(X1_matrix.flat, X2_matrix.flat)]
# mySpace = DiscreteSpace(X1_matrix = X1_matrix, X2_matrix = X2_matrix, U = U, Flat_board = Flat_board)


# #discretize the states and control space
# Dmn_row = len(mySpace.Flat_board)
# Dmn_U = mySpace.U.shape[0]
# myX1matrix = np.copy(mySpace.X1_matrix)
# myX2matrix = np.copy(mySpace.X2_matrix)

# cov = myPendulum.sigma.dot(myPendulum.sigma.T) * myMDP.delta_t
# Pr_transition = np.zeros((Dmn_U, Dmn_row, Dmn_row))
# stage_cost_table = np.zeros((Dmn_U, Dmn_row))
# #stage cost 
# stage_cost = lambda x1, u: (1. - np.exp(myPendulum.k * (np.cos(x1)-1)) + myPendulum.r*u**2./2) * myMDP.delta_t
# #prepare for f(X,U) fucntion, with single u input
# f1 = myX2matrix
# f2 = myPendulum.a * np.sin(myX1matrix) - myPendulum.b * myX2matrix
# myX1 = np.array(mySpace.Flat_board)[:,0]

# for u_i in range(Dmn_U):
#     u = mySpace.U[u_i, 0]
#     # calculate stage cost table
#     stage_cost_table[u_i, :] = stage_cost(myX1, u)
    
#     # implement state updates, X + f(X, U)*delta_t
#     f2_u = f2 + u
#     X1means = mySpace.X1_matrix + f1 * myMDP.delta_t # angle should be in the interval [-pi, pi]
#     X1means = (X1means + math.pi) % (2 * math.pi) - math.pi

#     X2means = mySpace.X2_matrix + f2_u * myMDP.delta_t 
#     # anglur velocity should be in interval[-vmax, vmax]
#     X2means[X2means < -myMDP.v_max] = -myMDP.v_max
#     X2means[X2means > myMDP.v_max] = myMDP.v_max
    
#     # following is my shifted state
#     myXmeans = [x for x in zip(X1means.flat, X2means.flat)]

#     for xmean_j, xmean in enumerate(myXmeans):
#         #find samples under each xmean of Xmeans, with gaussian distribution
#         pr_shiftpts = multivariate_normal.pdf(mySpace.Flat_board, mean=xmean, cov=cov)
#         threshold =  pr_shiftpts.max() * 0.6
#         #threshold =  multivariate_normal.pdf(xmean, mean=xmean, cov=cov) * 0.6
#         ind = np.array(pr_shiftpts) > threshold
        
#         pr_myXsamples = np.array(pr_shiftpts)[ind]
#         # normalize these samples with corresponding probability so that sum of them is 1
#         normalized_pr_myXsamples = pr_myXsamples / pr_myXsamples.sum()
#         Pr_transition[u_i, xmean_j, :][ind] = normalized_pr_myXsamples


class optimal ():
    def __init__(self, transition, cost, Dmn_U, Dmn_row):
        self.P = transition
        self.C = cost
        self.A = Dmn_U
        self.S = Dmn_row
        
    def _clean(self):
        self.value = np.zeros((self.S, 1))
        self.J = np.zeros((self.S, 1))
        self.policy = None
        self.discount = 0.9
    
    def _bellmanOperator(self, V):
        Q = np.zeros((self.A, self.S)) #A, S
        for aa in range(self.A):
            Q[aa] = self.C[aa].squeeze() + (self.discount * (self.P[aa]).dot(V)).squeeze()
            #1* 10000 = 1* 10000 + 10000* 10000 dot 10000 * 1 
        policy = Q.argmin(axis=0) #1*10000 in 1D in index
        value = Q.min(axis=0)[:, None] #1*10000 in 1D
        return policy, value
        
    def ValueIteration(self):
        self._clean()
        delta = None
        count = 0
        while True:
            count += 1
            V = self.value.copy()
            self.policy, self.value = self._bellmanOperator(V)
            delta = abs(self.value - V)
            if delta.all() < (1-self.discount)/self.discount:
                break
        print ("number of value iteration", count)
                
    def PolicyIteration(self):
        #initialize
        self._clean()
        delta = None
        policy_idx = np.random.choice(self.A, self.S)
        value_idx = np.arange(self.S)
        # initialize random policy 
        self.policy = np.random.choice(self.A, self.S)
        count = 0
        while True:
            count += 1
            # after break, we have best policy for each states, and value for each states
            g = self.C[self.policy, value_idx][:, None] #1 * 10000s
            P = np.zeros((self.S, self.S))
            for i in range(self.S):
                P[i, :] = self.P[self.policy[i], i, :]
            I_mat = np.eye(self.S)
            self.J = (np.linalg.inv(I_mat - self.discount * P)).dot(g)
            temp_policy, self.Value = self._bellmanOperator(self.J)
            #print (np.sum(np.abs(temp_policy - self.policy)) )
            if np.sum(np.abs(temp_policy - self.policy))<=3 :
                break
            else:
                 self.policy = temp_policy
        print ("number of Policy Iteration: ", count)



# myVI = optimal(Pr_transition, stage_cost_table, Dmn_U, Dmn_row)  
# myVI.ValueIteration()

# myPI = optimal(Pr_transition, stage_cost_table, Dmn_U, Dmn_row)  
# myPI.PolicyIteration()



# optimal_policy = myVI.policy.reshape(len(X2), len(X1))
# pos = np.array([math.pi, 0.0])
# pos_lst = []
# intpl = scipy.interpolate.interp2d\
# (np.arange(len(X1)), np.arange(len(X2)), optimal_policy, kind='linear')

# for i in range(0, 1000):
#     pos_lst.append(pos)
#     x1_id = (pos[0] + myMDP.theta_max) / (2. * myMDP.theta_max/(len(X1) - 1))
#     x2_id = (pos[1] + myMDP.v_max) / (2. * myMDP.v_max/(len(X2) - 1))
#     policy_idx = intpl(x1_id, x2_id) 
    
#     u = -myMDP.u_max + policy_idx * (2. * myMDP.u_max/ (len(U) - 1))
#     dw = np.random.multivariate_normal(np.array([0, 0]), \
#             np.array([[myMDP.delta_t, 0], [0, myMDP.delta_t]]), 1)[0]
    
#     fx_uu = np.array([pos[1], myPendulum.a * np.sin(pos[0]) - myPendulum.b * np.cos(pos[1]) + u])
#     pos = pos + myMDP.delta_t * fx_uu + (myPendulum.sigma.dot(dw)).flatten()
#     pos[0] = (pos[0] + math.pi) % (2 * math.pi) - math.pi
#     pos[1] = max(pos[1], -myMDP.v_max)
#     pos[1] = min(pos[1], myMDP.v_max)


# np.save('./src/myxuV.npy', np.array(pos_lst))
# np.save('./src/myxuP.npy', np.array(pos_lst))



if __name__ == "__main__":
	#initialize paramters settings
	MDPConstants = namedtuple('MDPConstants', ['delta_t','n_1', 'n_2', 'n_u', 'theta_max', 'v_max', 'u_max'])
	#theta is for X1, angular velocity is for X2, control is for u
	PendulumConstants = namedtuple('PendulumConstants', ['a', 'b', 'sigma', 'k', 'r', 'gamma'])
	#the constants that describe the pendulum
	DiscreteSpace = namedtuple('DiscreteSpace', ['X1_matrix', 'X2_matrix', 'U', 'Flat_board'])
	#The space describe the state space and control space

	myMDP = MDPConstants(delta_t=0.1, n_1=80, theta_max= math.pi, v_max=8, u_max=8, n_2=20, n_u=50)
	myPendulum = PendulumConstants(a=1, b=0, sigma=np.eye(2) * 0.1, k=1, r=0.01, gamma=0.3)

	X1 = np.linspace(-myMDP.theta_max, myMDP.theta_max, myMDP.n_1) # one D array for X1
	X2 = np.linspace(-myMDP.v_max, myMDP.v_max, myMDP.n_2) # one D array for X2

	X1_matrix, X2_matrix = np.meshgrid(X1, X2) #create the state space for X meshgrid format
	U = np.linspace(-myMDP.u_max, myMDP.u_max, myMDP.n_u)[:, None] # one D array for control U
	# where row number is for X2, column number is for X1
	Flat_board = [i for i in zip(X1_matrix.flat, X2_matrix.flat)]
	mySpace = DiscreteSpace(X1_matrix = X1_matrix, X2_matrix = X2_matrix, U = U, Flat_board = Flat_board)


	#discretize the states and control space
	Dmn_row = len(mySpace.Flat_board)
	Dmn_U = mySpace.U.shape[0]
	myX1matrix = np.copy(mySpace.X1_matrix)
	myX2matrix = np.copy(mySpace.X2_matrix)

	cov = myPendulum.sigma.dot(myPendulum.sigma.T) * myMDP.delta_t
	Pr_transition = np.zeros((Dmn_U, Dmn_row, Dmn_row))
	stage_cost_table = np.zeros((Dmn_U, Dmn_row))
	#stage cost 
	stage_cost = lambda x1, u: (1. - np.exp(myPendulum.k * (np.cos(x1)-1)) + myPendulum.r*u**2./2) * myMDP.delta_t
	#prepare for f(X,U) fucntion, with single u input
	f1 = myX2matrix
	f2 = myPendulum.a * np.sin(myX1matrix) - myPendulum.b * myX2matrix
	myX1 = np.array(mySpace.Flat_board)[:,0]

	for u_i in range(Dmn_U):
	    u = mySpace.U[u_i, 0]
	    # calculate stage cost table
	    stage_cost_table[u_i, :] = stage_cost(myX1, u)
	    
	    # implement state updates, X + f(X, U)*delta_t
	    f2_u = f2 + u
	    X1means = mySpace.X1_matrix + f1 * myMDP.delta_t # angle should be in the interval [-pi, pi]
	    X1means = (X1means + math.pi) % (2 * math.pi) - math.pi

	    X2means = mySpace.X2_matrix + f2_u * myMDP.delta_t 
	    # anglur velocity should be in interval[-vmax, vmax]
	    X2means[X2means < -myMDP.v_max] = -myMDP.v_max
	    X2means[X2means > myMDP.v_max] = myMDP.v_max
	    
	    # following is my shifted state
	    myXmeans = [x for x in zip(X1means.flat, X2means.flat)]

	    for xmean_j, xmean in enumerate(myXmeans):
	        #find samples under each xmean of Xmeans, with gaussian distribution
	        pr_shiftpts = multivariate_normal.pdf(mySpace.Flat_board, mean=xmean, cov=cov)
	        threshold =  pr_shiftpts.max() * 0.6
	        #threshold =  multivariate_normal.pdf(xmean, mean=xmean, cov=cov) * 0.6
	        ind = np.array(pr_shiftpts) > threshold
	        
	        pr_myXsamples = np.array(pr_shiftpts)[ind]
	        # normalize these samples with corresponding probability so that sum of them is 1
	        normalized_pr_myXsamples = pr_myXsamples / pr_myXsamples.sum()
	        Pr_transition[u_i, xmean_j, :][ind] = normalized_pr_myXsamples


	myVI = optimal(Pr_transition, stage_cost_table, Dmn_U, Dmn_row)  
	myVI.ValueIteration()
	optimal_policy = myVI.policy.reshape(len(X2), len(X1))
	pos = np.array([math.pi, 0.0])
	pos_lst = []
	intpl = scipy.interpolate.interp2d\
	(np.arange(len(X1)), np.arange(len(X2)), optimal_policy, kind='linear')

	for i in range(0, 1000):
	    pos_lst.append(pos)
	    x1_id = (pos[0] + myMDP.theta_max) / (2. * myMDP.theta_max/(len(X1) - 1))
	    x2_id = (pos[1] + myMDP.v_max) / (2. * myMDP.v_max/(len(X2) - 1))
	    policy_idx = intpl(x1_id, x2_id) 
	    
	    u = -myMDP.u_max + policy_idx * (2. * myMDP.u_max/ (len(U) - 1))
	    dw = np.random.multivariate_normal(np.array([0, 0]), \
	            np.array([[myMDP.delta_t, 0], [0, myMDP.delta_t]]), 1)[0]
	    
	    fx_uu = np.array([pos[1], myPendulum.a * np.sin(pos[0]) - myPendulum.b * np.cos(pos[1]) + u])
	    pos = pos + myMDP.delta_t * fx_uu + (myPendulum.sigma.dot(dw)).flatten()
	    pos[0] = (pos[0] + math.pi) % (2 * math.pi) - math.pi
	    pos[1] = max(pos[1], -myMDP.v_max)
	    pos[1] = min(pos[1], myMDP.v_max)
	np.save('./src/myxuV.npy', np.array(pos_lst))


	myPI = optimal(Pr_transition, stage_cost_table, Dmn_U, Dmn_row)  
	myPI.PolicyIteration()
	optimal_policy = myPI.policy.reshape(len(X2), len(X1))
	pos = np.array([math.pi, 0.0])
	pos_lst = []
	intpl = scipy.interpolate.interp2d\
	(np.arange(len(X1)), np.arange(len(X2)), optimal_policy, kind='linear')
	for i in range(0, 1000):
	    pos_lst.append(pos)
	    x1_id = (pos[0] + myMDP.theta_max) / (2. * myMDP.theta_max/(len(X1) - 1))
	    x2_id = (pos[1] + myMDP.v_max) / (2. * myMDP.v_max/(len(X2) - 1))
	    policy_idx = intpl(x1_id, x2_id) 
	    
	    u = -myMDP.u_max + policy_idx * (2. * myMDP.u_max/ (len(U) - 1))
	    dw = np.random.multivariate_normal(np.array([0, 0]), \
	            np.array([[myMDP.delta_t, 0], [0, myMDP.delta_t]]), 1)[0]
	    
	    fx_uu = np.array([pos[1], myPendulum.a * np.sin(pos[0]) - myPendulum.b * np.cos(pos[1]) + u])
	    pos = pos + myMDP.delta_t * fx_uu + (myPendulum.sigma.dot(dw)).flatten()
	    pos[0] = (pos[0] + math.pi) % (2 * math.pi) - math.pi
	    pos[1] = max(pos[1], -myMDP.v_max)
	    pos[1] = min(pos[1], myMDP.v_max)
	    
	np.save('./src/myxuP.npy', np.array(pos_lst))



