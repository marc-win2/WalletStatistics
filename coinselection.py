# Created by Marc Winstel on July 14, 2025
import numpy as np
from math import floor

from sympy import det
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
        self.mode = distMode # can be "grandcanonical" or "canonical", or "uniform"
        self.warnaboutZeroProbabilities = False  # Flag to warn about zero probabilities
        self.muArray = []
        self.rng = initializeRandomNumGenerator()

        self.preComputedProbabilities = [0.0]* len(self.tBucketBounds)  # Precomputed probabilities for each bucket bound  
        for i, bound in enumerate(self.tBucketBounds):
            self.preComputedProbabilities[i] = self.compProbabilityForBucket(i) # precompute the probabilities for the bucket bounds, where the upper bound is used to compute the probability for the bucket

        print(self.preComputedProbabilities)


        if self.mode == "grandcanonical":
            self.setGrandCanonical()

        if self.mode == "canonical":
            self.setCanonical()
        if self.mode == "uniform":
            self.setUniform()
        
    def returnBucketProbabilitiesForFixedWalletState(self, tokenNoPerBucket):
        """
        Pick a bucket based on the precomputed probabilities. Set Probabilities to zero for empty buckets.
        """
        probabilities = [0.0]* len(self.tBucketBounds)
        for i, bucket in enumerate(self.tBucketBounds):
            if tokenNoPerBucket[i] != 0: ## return zero probability for empty buckets
                probabilities[i] = self.compProbabilityForBucket(i)



        intBoundsForDrawing = []
        summing = 0.0
        for i, prob in enumerate(probabilities):
            summing += probabilities[i]
            intBoundsForDrawing.append(summing)

        return probabilities, intBoundsForDrawing



                
        


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

        for token in tokenSet:
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
    
    def setGrandCanonical(self, muValueGlobal=0.0):
        """
        Set the mode to grandcanonical.
        """
        self.mode = "grandcanonical"
        
        # Initialize betaMuArray based on the tBucketBounds
        self.muArray = [muValueGlobal for i in range(len(self.tBucketBounds))]
    
        
        # Set a default value for beta if not already set
        if self.beta == 0.0:
            print("Warning: beta is zero, grandcanonical mode, setting to 1.0.")
            self.beta = 1.0

    def setGrandCanonical(self, muValueArray = None):
        """
        Set the mode to grandcanonical with a specific chemical potential array.
        """
        self.mode = "grandcanonical"
        if muValueArray is not None:
            self.muArray = muValueArray
        else:
            self.muArray = [0.0 for i in range(len(self.tBucketBounds))]

        if self.beta == 0.0:
            print("Warning: beta is zero, grandcanonical mode, setting to 1.0.")
            self.beta = 1.0
    
    def setMuArray(self, muValueArray):
        """
        Set the chemical potential array for grandcanonical mode.
        """
        if self.mode != "grandcanonical":
            raise ValueError("Cannot set muArray in non-grandcanonical mode.")
        self.muArray = muValueArray

    def setCanonical(self):
        """
        Set the mode to canonical.
        """
        self.mode = "canonical"
        self.muArray = [0.0] * len(self.tBucketBounds)  # Reset muArray for canonical mode
        if self.beta == 0.0:
            print("Warning: beta is zero, corresponds to uniform distribution. Set to mode uniform.")
            self.setUniform()


    def setUniform(self):
        """
        Set the mode to uniform.
        """
        self.mode = "uniform"
        self.beta = 0.0
        self.muArray = None  # Reset muArray for uniform mode
        
    def setBeta(self, beta):
        """
        Set the inverse temperature (beta).
        """
        self.beta = beta
        #if self.mode == "grandcanonical":    
    

    def compProbabilityForBucket(self, bucketIndex):
        """
        Compute the probability for a given bucket index.
        """
        if bucketIndex < 0 or bucketIndex >= len(self.tBucketBounds):
            raise IndexError("Bucket index out of bounds.")
        
        if self.mode == "grandcanonical":
            mu = self.muArray[bucketIndex]
            return BoltzmannDistribution(self.tBucketBounds[bucketIndex], self.beta, mu)
        
        elif self.mode == "canonical":
            return BoltzmannDistribution(self.tBucketBounds[bucketIndex], self.beta, 0.0)
        
        elif self.mode == "uniform":
            return 1.0
        else:
            raise NotImplementedError("Only grandcanonical, canonical, and uniform modes are implemented.")


   


    def probabilityGrandCanonical(self, tokenValue):
        """
        Compute the distribution for a given token value.
        """
        if tokenValue < 0.0 or tokenValue > self.tBucketBounds[-1]:
            return 0.0

        bucketIndex = np.searchsorted(self.tBucketBounds, tokenValue, side='left')
        if bucketIndex < 0 or bucketIndex >= len(self.muArray):
            return 0.0

        mu = self.muArray[bucketIndex] 
        energy = tokenValue
        return BoltzmannDistribution(energy, self.beta, mu)
    
    def probabilityCanonical(self, tokenValue):
        """
        Compute the canonical distribution for a given token value.
        """
        if tokenValue < 0.0 or tokenValue > self.tBucketBounds[-1]:
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
        
    
 
