
import numpy as np
import matplotlib.pyplot as plt
from hill_automobile import HillAutomobile
import copy

def save_fig(data_list, filepath="./my_plot.png",x_label="X", y_label="Y", x_range=(0, 1), y_range=(0,1), color="red",  grid=True):
    if(len(data_list) <=1):
        return
    data_arr = np.array(data_list)
    time_arr = 1 + np.arange(data_arr.size)
    mu = np.divide(np.cumsum(data_arr),time_arr)
    std = np.sqrt(np.divide(np.sum(np.square(np.triu(data_arr[:,np.newaxis] - mu)),axis=0),time_arr))
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
    ax.grid(grid)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(time_arr-1,mu, color,  alpha=1.0)
    ax.fill_between(time_arr-1,mu+std, mu-std, facecolor=color, alpha=0.5)
    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)

class TemporalDifference():
    def __init__(self):
        self.N_actions = 3
        self.N_bins = 12  #plus is the number of disctrized states total states 11
        self.epsilon = 0.1
        self.gamma = 0.9999
        self.alpha = 0.001

        self.position_space = np.linspace(-1.2, 0.5, self.N_bins-1, endpoint=False)
        self.velocity_space = np.linspace(-1.5, 1.5, self.N_bins-1, endpoint=False)
  
    def initUtilitymatrix(self):
        #initialize the policy randomly, 10*10
        self.policy_matrix = np.random.randint(low=0,high=self.N_actions,size=(self.N_bins,self.N_bins))
        self.Q_matrix = np.zeros((self.N_bins,self.N_bins, self.N_actions))
        #visited_counter = np.zeros((self.N_bins,self.N_bins, self.N_actions))
    
    def epsilonGreedypolicy(self, s):
        action_idx = int(self.policy_matrix[s])
        greedy_prob = 1 - self.epsilon + self.epsilon/self.N_actions
        not_greedy_prob = self.epsilon/self.N_actions
        weight = np.full((self.N_actions),not_greedy_prob)
        weight[action_idx] = greedy_prob
        selected_action = np.random.choice(self.N_actions,1,p=weight)
        return int(selected_action)
        
    def updateQmatrix(self,st,at,st1,at1,cost):
        Qst_at = self.Q_matrix[st[0],st[1],at]
        Qst1_at1 = self.Q_matrix[st1[0],st1[1],at1]
        self.Q_matrix[st[0],st[1],at] = Qst_at + self.alpha * (cost+self.gamma*Qst1_at1-Qst_at)
        
    def updatePolicymatrix(self,st):
        optimal_action = np.argmin(self.Q_matrix[st[0],st[1],:])
        self.policy_matrix[st[0],st[1]] = optimal_action
    
    @staticmethod
    def print_policy(policy_matrix):
        counter = 0
        shape = policy_matrix.shape
        policy_string = ""
        for row in range(shape[0]):
            for col in range(shape[1]):
                if policy_matrix[row,col] == 0: policy_string += " <  "
                elif policy_matrix[row,col] == 1: policy_string += " O  "
                elif policy_matrix[row,col] == 2: policy_string += " >  "
                counter += 1
            policy_string += '\n'
        print(policy_string)


if __name__ == '__main__':        
    episodes = 35000
    cost_list = [] # track the cost in each episode
    step_list = [] # track the number of steps per episode
    myCar = HillAutomobile(mass=0.2, friction=0.3, dt=0.1)
    myTD = TemporalDifference()
    myTD.initUtilitymatrix()
    for episode in range(episodes):
        print(episode)
        obs_st = myCar.reset(exploring_starts=False)
        st = (np.digitize(obs_st[0],myTD.position_space),np.digitize(obs_st[1],myTD.velocity_space))
        at = myTD.epsilonGreedypolicy(st)
        cumulative_cost = 0
        for step in range(100):
            obs_st1, cost, done = myCar.update(at)
            st1 = (np.digitize(obs_st1[0],myTD.position_space),np.digitize(obs_st1[1],myTD.velocity_space))
            at1 = myTD.epsilonGreedypolicy(st1)
            myTD.updateQmatrix(st,at,st1,at1,cost)
            myTD.updatePolicymatrix(st)
            st = copy.deepcopy(st1)
            at = copy.deepcopy(at1)
            cumulative_cost +=cost
            if done: break
        cost_list.append(cumulative_cost)
        step_list.append(step)
    #exit()
    # Visualize
    myCar.render()
    # ouput graph for cost
    print("Saving the cost plot in: ./cost.png")
    save_fig(cost_list, filepath="./cost.png",          x_label="Episode", y_label="Cost",          x_range=(0, len(cost_list)), y_range=(-1.5,1.5), color="red", grid=True)
    # output graph for step
    print("Saving the step plot in: ./step.png")
    save_fig(step_list, filepath="./step.png",       x_label="Episode", y_label="Steps",       x_range=(0, len(step_list)), y_range=(-0.1,130), color="blue", grid=True)
        
    myTD.print_policy(myTD.policy_matrix)
    print("Complete!")
