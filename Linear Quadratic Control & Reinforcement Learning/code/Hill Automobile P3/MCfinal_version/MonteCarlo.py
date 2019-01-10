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

class MonteCarlo():
    def __init__(self):
        self.N_actions = 3
        self.N_bins = 24  #plus is the number of disctrized states total states 11
        self.epsilon = 0.3
        self.discount = 0.995
        self.position_space = np.linspace(-1.2, 0.5, self.N_bins-1, endpoint=False)
        self.velocity_space = np.linspace(-1.5, 1.5, self.N_bins-1, endpoint=False)
  
    def initUtilitymatrix(self):
        #initialize the policy randomly, 10*10
        self.policy_matrix = np.random.randint(low=0,high=self.N_actions,size=(self.N_bins,self.N_bins))
        self.Q_matrix = np.zeros((self.N_bins,self.N_bins, self.N_actions))
        self.visited_counter = np.zeros((self.N_bins,self.N_bins, self.N_actions))
        
    def generateEpisode(self, obs_st, myCar):
        st_lst = []
        at_lst = []
        cost_lst = []
        for step in range(200):
            st = (np.digitize(obs_st[0],self.position_space),np.digitize(obs_st[1],self.velocity_space))
            at = int(self.policy_matrix[st])
            obs_st, cost, done = myCar.update(at) #  obs_st = obs_st1
            st_lst.append(st)
            at_lst.append(at)
            cost_lst.append(cost)
            if done:break
        return  st_lst, at_lst, cost_lst, step

    def updateQmatrix(self, states, actions, costs):
        checking_first = np.zeros((self.N_bins,self.N_bins, self.N_actions))
        converted_costs = [(self.discount**j)*cost for j, cost in enumerate(costs)]
        for i, sa in enumerate(zip(states, actions)):
            s_a = (sa[0][0], sa[0][1], sa[1])
            if checking_first[s_a] == 0:
                checking_first[s_a] +=1
                self.visited_counter[s_a] +=1
                Gs_a = sum(converted_costs[i:])/self.discount**i
                alpha = 1/(self.visited_counter[s_a]+1)
                self.Q_matrix[s_a] = self.Q_matrix[s_a]+alpha*(Gs_a - self.Q_matrix[s_a])
                
    def epsilonGreedypolicy(self, s):
        action_idx = np.argmin(self.Q_matrix[s[0],s[1],:])
        greedy_prob = 1 - self.epsilon + self.epsilon/self.N_actions
        not_greedy_prob = self.epsilon/self.N_actions
        weight = np.full((self.N_actions),not_greedy_prob)
        weight[action_idx] = greedy_prob
        selected_action = np.random.choice(self.N_actions,1,p=weight)
        return int(selected_action)

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
    myMC = MonteCarlo()
    myMC.initUtilitymatrix()

    for episode in range(episodes):
        print(episode)
        if episode < episodes-1:
            obs_st = myCar.reset(exploring_starts=True)
        if episode == episodes-1:
            obs_st = myCar.reset(exploring_starts=False) 
        states, actions, costs, step = myMC.generateEpisode(obs_st, myCar)
        cost_list.append(sum(costs))
        step_list.append(step)
        myMC.updateQmatrix(states, actions, costs)
        for s in states:
            optimal_action = myMC.epsilonGreedypolicy(s)
            myMC.policy_matrix[s] = optimal_action
    #exit() 
    myCar.render()
    print("Saving the cost plot in: ./cost.png")
    save_fig(cost_list, filepath="./cost.png",          x_label="Episode", y_label="Cost",          x_range=(0, len(cost_list)), y_range=(-1.5,1.5), color="red", grid=True)
    # output graph for step
    print("Saving the step plot in: ./step.png")
    save_fig(step_list, filepath="./step.png",       x_label="Episode", y_label="Steps",       x_range=(0, len(step_list)), y_range=(-0.1,130), color="blue", grid=True)
        
    myMC.print_policy(myMC.policy_matrix)
    print("Complete!")

    


