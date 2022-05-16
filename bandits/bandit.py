import numpy as np
import pymc3 as pm
from scipy.stats import truncnorm

class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0
        self.optimal_mean = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0
        self.optimal_mean = 0
        

    def pull(self, action):
        return 0, True, self.action_values[action]


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)
        self.optimal_mean = np.max(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal,self.action_values[action])
        
        
        
class TruncatedGaussianBandit(MultiArmedBandit):
    """
    Truncated Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0.5, sigma=1,sigma_means = 0.4, lower=0, upper=1):
        super(TruncatedGaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.sigma_means = sigma_means
        self.lower = lower
        self.n = 1
        self.upper = upper
        self.reset()

    def reset(self):
        self.dist = pm.TruncatedNormal.dist(mu=self.mu,sigma=self.sigma_means,upper=self.upper,lower=self.lower)
        self.action_values = self.dist.random(size = self.k)
        self.arms = np.array([pm.TruncatedNormal.dist(mu=mean,sigma=self.sigma,lower=self.lower,upper=self.upper) for mean in self.action_values])
        
        # self.action_values = [1/i for i in range(1,self.k+1)]
        self.optimal = np.argmax(self.action_values)
        self.optimal_mean = np.max(self.action_values)

    def pull(self, action):
        reward = self.arms[action].random()
        action_value = self.action_values[action]
        optimal = action == self.optimal
        
        return (reward, optimal,action_value)


class BinomialBandit(MultiArmedBandit):
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """
    def __init__(self, k, n, p=None, t=None):
        super(BinomialBandit, self).__init__(k)
        self.n = n
        self.p = p
        self.t = t
        self.model = pm.Model()
        with self.model:
            self.bin = pm.Binomial('binomial', n=n*np.ones(k, dtype=np.int),
                                   p=np.ones(k)/n, shape=(1, k), transform=None)
        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = np.random.uniform(size=self.k)
        else:
            self.action_values = self.p
        self.bin.distribution.p = self.action_values
        if self.t is not None:
            self._samples = self.bin.random(size=self.t).squeeze()
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)
        self.optimal_mean = np.max(self.action_values)
        

    def pull(self, action):

   
        return self.sample[action], action == self.optimal,self.action_values[action]

    @property
    def sample(self):
        if self._samples is None:
            return self.bin.random()
        else:
            if self._cursor > self._samples.shape[1]:
                self._cursor = 0
            
            val = self._samples[self._cursor]
            self._cursor += 1
            return val
        
        
        


class BernoulliBandit():
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This is
    the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss event,
    such as if a user clicks on a headline, ad, or recommended product.
    """
    def __init__(self, k, p_array = None):
        self.k = k
        self.n = 1
        if p_array is None:
            self.p_array = np.array([1/i for i in range(1,k+1)])
        
        else:
            self.p_array = p_array
            
            
        self.optimal = np.argmax(self.p_array)
        self.optimal_mean = self.p_array[self.optimal]
        
    
    def reset(self,p_array = None,k = None):

        if k is None:
            k = self.k
            
        if p_array is None:
            p_array = self.p_array
            
        self.__init__(self.k,p_array)
        
        
        
    def pull( self, i ):
        #i is which arm to pull
        # print(self.p_array)
        # print(self.optimal)
        
        return np.random.rand() < self.p_array[i], i == self.optimal,self.p_array[i]
    
    def __len__(self):
        return len(self.p_array)
