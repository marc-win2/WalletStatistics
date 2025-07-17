# Created by Marc Winstel on July 14, 2025
import numpy as np




def BoltzmannDistribution(energy, beta, mu=0.0):
    """
    Calculate the Boltzmann distribution for a given energy and inverse temperature (beta).
    """
    return np.exp(-beta * (energy - mu)) 




class CoinSelectionDistribution:
    """
    Class to handle coin selection distribution calculations.
    """
    def __init__(self, beta, tokenDenominationBuckets):
        self.beta = beta  # Inverse temperature
        self.tBucketBounds = tokenDenominationBuckets
        self.muArray = []
        self.expn = []
        self.mode = "grandcanonical"
        
        
        
        self.fixTokenNoGlobally(10.0)  # Example value, can be adjusted        
            
    
    def compDenominatorDistribution(self, tokenSet):
        """
        Computes  \sum_t exp(-beta * t.value) for the distribution of a set of tokens
        """
        sum = 0.0
        
        for i, token in enumerate(tokenSet):
            val = token.value()
            sum += self.compProbability(val)
        return sum 
    
    def compDistribution(self, tokenSetInWallet, transactionValue):
        """
        Compute the coin selection distribution for a set of tokens, given a fixed transaction value.
        """
        tokenSet = [token for token in tokenSetInWallet if (token.value > 0.0 and token.value <= transactionValue)]
        denominator = self.compDenominatorDistribution(tokenSet, transactionValue)
        probabilities = [] # list of all p(t.value) = exp(-beta *t.value) / denominator, for all token in tokenSet (i.e., only those with value > 0.0 and <= transactionValue)
 
        
        for i, token in enumerate(tokenSet):
            val = token.value()
            prob = self.compProbability(val) / denominator
            probabilities.append(prob)
            
        return probabilities, tokenSet, denominator
            
    
    
    
    def setCanonical(self):
        """
        Set the mode to canonical.
        """
        self.mode = "canonical"
        self.muArray = [0.0] * len(self.tBucketBounds)
        self.expn = [] 
        
    def setBeta(self, beta):
        """
        Set the inverse temperature (beta).
        """
        self.beta = beta
        if self.mode == "grandcanonical":    
            self.muArray = [np.log(expn) / beta + t for expn, t in zip(self.expn, self.tBucketBounds)]

    
    def fixTokenNoGlobally(self, tokenNoFixed):
        """
        Set a fixed number of tokens globally.
        """
        self.expn = [tokenNoFixed] * len(self.tBucketBounds)
        self.muArray = [np.log(expn) / self.beta + t for expn, t in zip(self.expn, self.tBucketBounds)]


    def probabilityGrandCanonical(self, tokenValue):
        """
        Compute the distribution for a given token value.
        """
        if tokenValue < self.tBucketBounds[0] or tokenValue >= self.tBucketBounds[-1]:
            return 0.0
        
        bucketIndex = np.searchsorted(self.tBucketBounds, tokenValue, side='right') 
        if bucketIndex < 0 or bucketIndex >= len(self.muArray):
            return 0.0
        
        mu = self.muArray[bucketIndex]
        energy = tokenValue
        return BoltzmannDistribution(energy, self.beta, mu)
    
    def probabilityCanonical(self, tokenValue):
        """
        Compute the canonical distribution for a given token value.
        """
        if tokenValue < self.tBucketBounds[0] or tokenValue >= self.tBucketBounds[-1]:
            return 0.0
        energy = tokenValue
        return BoltzmannDistribution(energy, self.beta, 0.0)
    
    
    def compProbability(self, tokenValue):
        """
        Compute the probability for a given token value.
        """
        if self.mode == "grandcanonical":
            return self.probabilityGrandCanonical(tokenValue)
        if self.mode == "canonical":
            return self.probabilityCanonical(tokenValue)
        else:
            raise NotImplementedError("Only canonical mode is implemented.")
        
    
 
