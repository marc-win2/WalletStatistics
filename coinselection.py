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
    def __init__(self, beta, tokenDenominationBuckets, distMode="canonical"):
        self.beta = beta  # Inverse temperature. should be reabsorbed into the real number generation until return value is computed
        self.tBucketBounds = tokenDenominationBuckets
        self.betaMuArray = []
        self.expn = []
        self.mode = distMode # can be "grandcanonical" or "canonical", or "uniform"
        self.warnaboutZeroProbabilities = False  # Flag to warn about zero probabilities

        self.rng = initializeRandomNumGenerator()

        
        if beta == 0.0:
            print("Warning: beta is zero, corresponds to uniform distribution. Set to mode uniform.")
            self.beta = 0.0
            self.mode = "uniform"
            self.betaMuArray = [0.0] * len(self.tBucketBounds)
            self.expn = []
        else:
            self.betaMuArray = [np.log(expn) + t for expn, t in zip(self.expn, self.tBucketBounds)]
        if self.mode == "grandcanonical":
            self.fixExpTokenNoGlobally(10.0)  # Example value, can be adjusted
        if self.mode == "canonical":
            self.setCanonical()
        if self.mode == "uniform":
            self.setUniform()


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
        the probabilities are computed as p(t.value) = exp(-beta * t.value)
        Denominator  contains \sum_t exp(-beta * t.value) for the distribution of a set of tokens (does not use the function removeTokensHigherThanTransactionValue to optimize computation time).
        intBoundsForUniformDrawing builds intervals such that a uniform random number can be drawn from [0, denominator] and the corresponding token can be selected.
        """
        tokenSet = tokenSetInWallet
        denominator = 0.0
        probabilities = [] # list of all p(t.value) = exp(-beta *t.value) for all token in tokenSet (i.e., only those with value > 0.0 and <= transactionValue)
        intBoundsForUniformDrawing = []

        for i, token in enumerate(tokenSet):
            val = token.value
            prob = self.compProbability(val) 
            probabilities.append(prob)
            denominator += prob
            intBoundsForUniformDrawing.append(denominator)
            
        if denominator == 0.0:
            if tokenSet == []:
                print("Warning: denominator is zero.")
                print("Warning: tokenSet is empty, returning empty distribution.")
                print("    tokenSet = ", tokenSet, "tokenSetInWallet = ", tokenSetInWallet, "transactionValue = ", transactionValue, "probs = ", probabilities, "intBoundsForUniformDrawing = ", intBoundsForUniformDrawing)
                return [], [], tokenSet
            else: 
                if self.warnaboutZeroProbabilities:
                    print("Warning: denominator is zero.")
                    print("TokenSet is not empty, but all probabilities are zero. Often, this happens when tokens have very high values. Return uniform distribution.")
                    print("   tokenSet = ", tokenSet, "tokenSetInWallet = ", tokenSetInWallet, "probs = ", probabilities)
                denominator = 1.0  # Set to 1.0 to avoid division by zero
                probabilities = [1.0 / len(tokenSet)] * len(tokenSet)
                intBoundsForUniformDrawing = [i / len(tokenSet) for i in range(1, len(tokenSet) + 1)]


        if tokenSet == []:
            print("Warning: tokenSet is empty, returning empty distribution.")
            print("   tokenSet = ", tokenSet, "tokenSetInWallet = ", tokenSetInWallet, "transactionValue = ", transactionValue, "probs = ", probabilities, "intBoundsForUniformDrawing = ", intBoundsForUniformDrawing)
            return [], [], tokenSet
        
        probabilities = [p / denominator for p in probabilities]  # Normalize probabilities
                    
        return probabilities, intBoundsForUniformDrawing
            
    
    
    
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
            value = - np.log(1.0 - uniformrandom[0]) 

        if setChangeFlag:
            self.mode = "grandcanonical"
            print("Mode changed to grandcanonical again within the function pickValueFromContinuousDistribution.")

        return value / b
    
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

    def pickValueFromContinuousDistributionWithVariableBeta(self, b):
        """
        Pick a token value from the continuous distribution exp(-beta * t.value) for t.value in [0, inf) with variable beta.
        """
        value = 0.0
        originalbeta = self.beta
        self.beta = b
        value = self.pickValueFromContinuousDistribution()
        self.beta = originalbeta  # Reset beta to its original value

        return value 
        
    def pickValueFromContinuousDistributionWithBetaAdjustment(self, value):
        """
        Pick a token value from the continuous distribution exp(-beta * t.value)with beta adjustment such that mean is equal to 0.1*value
        """
        beta = 1.0 / (0.1 * value)  # Adjust beta such that the mean is equal to 0.1 * value
        return self.pickValueFromContinuousDistributionWithVariableBeta(beta)
    
    def setCanonical(self):
        """
        Set the mode to canonical.
        """
        self.mode = "canonical"
        self.betaMuArray = [0.0] * len(self.tBucketBounds)
        self.expn = [] 

    def setUniform(self):
        """
        Set the mode to uniform.
        """
        self.mode = "uniform"
        self.betaMuArray = [0.0] * len(self.tBucketBounds)
        self.expn = []
        self.beta = 0.0
        
    def setBeta(self, beta):
        """
        Set the inverse temperature (beta).
        """
        self.beta = beta
        if self.mode == "grandcanonical":    
            self.fixExpTokenNoGlobally(10.0)  # Example value, can be adjusted
    
    def fixExpTokenNoGlobally(self, tokenNoFixed):
        """
        Set a fixed number of tokens globally.
        """
        if self.mode != "grandcanonical":
            self.expn = []
            self.betaMuArray = [0.0] * len(self.tBucketBounds)

        else:
            self.expn = [tokenNoFixed] * len(self.tBucketBounds)
            self.betaMuArray = [np.log(expn) + t for expn, t in zip(self.expn, self.tBucketBounds)]


    def probabilityGrandCanonical(self, tokenValue):
        """
        Compute the distribution for a given token value.
        """
        if tokenValue < 0.0 or tokenValue >= self.tBucketBounds[-1]:
            return 0.0
        
        bucketIndex = np.searchsorted(self.tBucketBounds, tokenValue, side='right') 
        if bucketIndex < 0 or bucketIndex >= len(self.betaMuArray):
            return 0.0
        
        mu = self.betaMuArray[bucketIndex] / self.beta
        energy = tokenValue
        return BoltzmannDistribution(energy, self.beta, mu)
    
    def probabilityCanonical(self, tokenValue):
        """
        Compute the canonical distribution for a given token value.
        """
        if tokenValue < 0.0 or tokenValue >= self.tBucketBounds[-1]:
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
        if self.mode == "uniform":
            return 0.1 # Uniform distribution, return a constant probability, use smaller value for the case of many tokens 
        else:
            raise NotImplementedError("Only canonical mode is implemented.")
        
    
 
