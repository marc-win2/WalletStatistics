# Created by Marc Winstel on July 14, 2025
import numpy as np
from math import floor
from transaction import initializeRandomNumGenerator, generateUniformFloats, generateGaussianFloats



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
        
        self.rng = initializeRandomNumGenerator()
        
        if beta == 0.0:
            print("Warning: beta is zero, currently not possible (should become a uniform distribution in general). Set to 1.0")
            self.beta = 1.0
        
        
        
        self.fixTokenNoGlobally(10.0)  # Example value, can be adjusted        
            
    
    def compDenominatorDiscDistribution(self, tokenSet):
        """
        Compute the denominator for the coin selection distribution.
        """
        sum = 0.0
        
        for i, token in enumerate(tokenSet):
            val = token.value()
            sum += self.compProbability(val)
        return sum 
    
    def removeTokensHigherThanTransactionValue(self, tokenSetInWallet, transactionValue):
        """
        Remove tokens from the wallet that have a value greater than the transaction value.
        """
        return [token for token in tokenSetInWallet if token.value <= transactionValue]
    
    def compDistributionDiscrSet(self, tokenSetInWallet, transactionValue):
        """
        Compute the coin selection distribution for a set of tokens, given a fixed transaction value.
        First, all tokens with value > 0.0 and <= transactionValue are selected.
        Then, the probabilities are computed as p(t.value) = exp(-beta * t.value)
        Denominator  contains \sum_t exp(-beta * t.value) for the distribution of a set of tokens (does not use the function removeTokensHigherThanTransactionValue to optimize computation time).
        intBoundsForUniformDrawing builds intervals such that a uniform random number can be drawn from [0, denominator] and the corresponding token can be selected.
        """
        tokenSet = self.removeTokensHigherThanTransactionValue(tokenSetInWallet, transactionValue)
        denominator = 0.0
        probabilities = [] # list of all p(t.value) = exp(-beta *t.value) for all token in tokenSet (i.e., only those with value > 0.0 and <= transactionValue)
        intBoundsForUniformDrawing = []
        
        for i, token in enumerate(tokenSet):
            val = token.value()
            prob = self.compProbability(val) 
            probabilities.append(prob)
            denominator += prob
            intBoundsForUniformDrawing.append(denominator)
            
        if denominator == 0.0:
            print("Warning: denominator is zero, returning empty distribution.")
            return [], [], tokenSet
        
        probabilities = [p / denominator for p in probabilities]  # Normalize probabilities
                    
        return probabilities, intBoundsForUniformDrawing, tokenSet
            
    
    
    
    def pickValueFromContinuousDistribution(self, b =None):
        """
        Pick a token value from the continuous distribution exp(-beta * t.value) for t.value in [0, inf) 
        """
        value = 0.0
        
        if b is None:
            b = self.beta


        setChangeFlag = False
        
        if self.mode == "grandcanonical":
            print("Warning: grandcanonical mode is not implemented  for the function pickValueFromContinuousDistribution. Use canonical mode instead.")     
            self.mode = "canonical"
            setChangeFlag = True
        
        if self.mode == "canonical":
            uniformrandom = generateUniformFloats(self.rng, 1, 0.0, 1.0)
            value = np.log(1 - b * uniformrandom) / (- b)

        if setChangeFlag:
            self.mode = "grandcanonical"
            print("Mode changed to grandcanonical again within the function pickValueFromContinuousDistribution.")

        return value 
    
    def pickValueFromContinuousDistributionWithinUpperBound(self, tValue ):
        """
        Pick a token value from the continuous distribution exp(-beta * t.value) for t.value in [0, tValue] 
        """
        value =  self.pickValueFromContinuousDistribution()
        
        if value > tValue:
            multInteger =floor(value / tValue)
            
            value = value - multInteger*tValue
        
        if value < 0.0:
            value = 0.0
        
        return value        
        
    
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
        
    
 
