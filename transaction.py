# Created by Marc Winstel on 14.07.25
import numpy as np 
import matplotlib.pyplot as plt

from scipy.integrate import quad


def initializeRandomNumGenerator(seed=None):
    """
    Initialize the random number generator with (possibly) given seed.
    """
    randomNumGen = None 
    if seed is None:
        randomNumGen = np.random.default_rng()
    else:
        randomNumGen = np.random.default_rng(seed)
    
    return randomNumGen


def generateUniformFloats(rng, n, lowBound = 0.0, highBound = 1.0):
    """
    Generate n uniformly distributed floats in the range [lowBound, highBound).
    """
    return rng.uniform(low=lowBound, high=highBound, size=n)


def generateGaussianFloats(rng, n, mean=0.0, stdDev=1.0):
    """
    Generate n Gaussian distributed floats with given mean and standard deviation.
    """
    return rng.normal(loc=mean, scale=stdDev, size=n)



class RandomTransactionGenerator:
    """
    Class to generate random transactions.
    """
    def __init__(self, seed=None):
        self.rng = initializeRandomNumGenerator(seed)
        self.maxAbsTransactionValue = 10**7 ### maximal, absolute transaction value
        
    def generateTransactionUniform(self):
        """
        Generate a random transaction with uniform distribution.
        """
        transactionValue = self.rng.uniform(-1.0*self.maxAbsTransactionValue, self.maxAbsTransactionValue)
        return transactionValue
    
    def generateTransationUniformPoitive(self):
        """
        Generate a random transaction with uniform distribution, but only positive values.
        """
        transactionValue = self.rng.uniform(0.0, self.maxAbsTransactionValue)
        return transactionValue
    
    def generateTransactionGaussian(self, stdDev=1.0, mean=0.0):
        """
        Generate a random transaction with Gaussian distribution.
        """
        transactionValue = self.rng.normal(loc=mean, scale=stdDev)
        # Ensure the transaction value is within the allowed range
        if np.abs(transactionValue) > self.maxAbsTransactionValue:
            transactionValue = np.sign(transactionValue)*( np.abs(transactionValue) - self.maxAbsTransactionValue )
        return transactionValue
    
    
    def generateNTransactionsGaussian(self, n, stdDev=1000.0, mean=0.0):
        """
        Generate n random transactions with Gaussian distribution.
        """
        # Ensure all transaction values are within the allowed range
        transactions = generateGaussianFloats(self.rng, n, mean, stdDev)
        transactions = np.clip(transactions, -self.maxAbsTransactionValue, self.maxAbsTransactionValue) # Ensure all transactions are within the allowed range
        transactions =  [ t for t in transactions if ((np.abs(t) < 1e-03) == False)] # Ensure no transaction is too small
        return transactions
    
    
    def generateMostTransactionsGaussianButWithUniformOutliers(self, n, stdDev, mean, outlierFraction=0.1):
        """
        Generate n random transactions with Gaussian distribution, but with a fraction of uniform outliers.
        """
        numOutliers = int(n * outlierFraction)
        numNormal = n - numOutliers
        
        normalTransactions = self.generateNTransactionsGaussian(numNormal, stdDev, mean)
        outlierTransactions = generateUniformFloats(self.rng, numOutliers, 0.95*self.maxAbsTransactionValue, self.maxAbsTransactionValue)
        
        transactions = np.concatenate((normalTransactions, outlierTransactions))
        self.rng.shuffle(transactions)
        
        return transactions
    
    
    def plotGaussianTransactionTestOne(self, n = 1000):
        """
        Generate and plot a test of random transactions.
        """
        
        transactions = self.generateNTransactionsGaussian(n)
        plt.hist(transactions, bins=100, density=False)
        plt.title("Histogram of Random Transactions")
        plt.xlabel("Transaction Value")
        plt.ylabel("Density")
        plt.grid()
        plt.show()
        return transactions
    
    def plotGaussianWithUniformOutliersTransactionTestOne(self, n = 1000, stdDev=10000.0, mean=-6000.0, outlierFraction=61e-04):
        """
        Generate and plot a test of random transactions with Gaussian distribution and uniform outliers.
        """
        
        transactions = self.generateMostTransactionsGaussianButWithUniformOutliers(n, stdDev, mean, outlierFraction)
        plt.hist(transactions, bins=100, density=False)
        plt.title("Histogram of Random Transactions with Uniform Outliers")
        plt.xlabel("Transaction Value")
        plt.ylabel("Density")
        plt.grid()
        plt.show()
        return transactions