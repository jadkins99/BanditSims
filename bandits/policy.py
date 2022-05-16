import numpy as np
import math

class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0
    
    
    
class ExploreFirstPolicy(Policy):
    def __init__(self, N,K):
        self.K = K
        self.N = N
        self.t = 1
    def __str__(self):
        return 'explore first policy'
    
    def choose(self, agent):
        if self.t < self.N*self.K:
            action = math.floor(self.t/self.N)
            self.t += 1
            return action
        
        else:
            action = np.argmax(agent.value_estimates)
            if len(agent.action_attempts) == 1:
                
             check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
             
            else:
                check = np.where(agent.value_estimates == agent.value_estimates[action])
            
            self.t += 1
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)

class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            
            if len(agent.value_estimates) <= 1:
                
                check = np.where(agent.value_estimates[0] == agent.value_estimates[0][action])
             
            else:
              
                
                check = np.where(agent.value_estimates == agent.value_estimates[action])
            
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random'


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c,num_trials):
        self.c = c
        self.T = num_trials

    def __str__(self):
        return 'UCB1'

    def choose(self, agent):
        
        exploration = 2*np.log(self.T) / agent.action_attempts
        exploration[np.isnan(exploration)] = 2*np.log(self.T)
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) == 1:
            return action
        else:
            return np.random.choice(check)


class SoftmaxPolicy(Policy):
    """
    The Softmax policy converts the estimated arm rewards into probabilities
    then randomly samples from the resultant distribution. This policy is
    primarily employed by the Gradient Agent for learning relative preferences.
    """
    def __str__(self):
        return 'SM'

    def choose(self, agent):
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]
