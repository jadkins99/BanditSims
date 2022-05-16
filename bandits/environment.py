import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from bandits.agent import BetaAgent


class Environment(object):
    def __init__(self, bandit, agents):
        self.bandit = bandit
        self.agents = agents
        # self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)
        regret =  np.zeros((trials, len(self.agents)))
        

        for _ in range(experiments):
            print(f"starting  {_}")
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    
                        
                    action = agent.choose()
                    reward, is_optimal,mean_reward = self.bandit.pull(action)
                    agent.observe(reward,action_taken = action)
                    
                    if t%10000 == 0:
                        print('we are at time: ',t)
                        print(agent.alpha)
                        print(agent.beta)
                        print(action)
                        print(reward)
                        
                        print(is_optimal)
                    scores[t, i] = reward
                    regret[t,i] = self.bandit.optimal_mean - mean_reward if t == 0 else self.bandit.optimal_mean - mean_reward + regret[t-1,i]
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments, regret / experiments

    def plot_reward(self, scores,label):
        sns.set_style('white')
        sns.set_context('talk')
      
        plt.xlabel('Time Step')
        
        
        
        
        plt.plot(scores,marker='.',linestyle='None')
        plt.ylabel(label)
        plt.legend(self.agents, loc=4)
    
    
    
    def plot_regret(self,regret,label):
        sns.set_style('white')
        sns.set_context('talk')
      
        plt.ylabel(label)

        plt.xlabel('Time Step')
        
        
        plt.plot(regret)
        plt.legend(self.agents, loc=4)
        
        
        sns.despine()
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        # plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            if type(agent) is not BetaAgent:
                for j, val in enumerate(agent.value_estimates):
                    axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)
            else:
                x = np.arange(0, 1, 0.001)
                y = np.array([stats.beta.pdf(x, a, b) for a, b in
                             zip(agent.alpha, agent.beta)])
                y /= np.max(y)
                for j, _y in enumerate(y):
                    axes[j].plot(x, _y, color=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()
