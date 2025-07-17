# Created by Marc Winstel on July 17, 2025
import numpy as np
from transaction import initializeRandomNumGenerator, generateUniformFloats 
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet


class SimulationHandler:
    """
    Class to handle simulation of coinselection.
    """
    def __init__(self, tokenDenominationBuckets):
        
        ### for transaction handling
        self.transactionSet = [10000.0]        
        self.currentTransactionIndex = 0
        self.transactionSetSize = len(self.transactionSet)
        self.coinCountPerTransaction = [0] * self.transactionSetSize
        self.globalTokenIndex = 0
        
        
        
    
        self.highThroughputWallet = Wallet()
        self.coinSelectionDistr = CoinSelectionDistribution(1.0, tokenDenominationBuckets)
        self.coinSelectionDistr.setCanonical()
        
        
        self.initiateWallet()  # Initialize the wallet with a set of tokens 
        
    
    def selectTokenFromDistribtion(self,transactionValue):
        """
        Select a token from the distribution based on the transaction value.
        """
        probs, sumOfProbs, redTokenSet = self.coinSelectionDistr.compDistributionDiscrSet(self.highThroughputWallet.tokens, transactionValue) # redTokenSet is the reduced token set with values <= transactionValue, sumOfProbs[i] contains the sum of probabilities for all tokens up to index i in redTokenSet, probs[i] contains the probability for the token at index i in redTokenSet
        
        
        rng = initializeRandomNumGenerator()
        randomFloat = generateUniformFloats(rng, 1, 0.0, sumOfProbs[-1]) # generate a random float in the range [0, sumOfProbs[-1]) where sumOfProbs[-1] is the sum of all probabilities
        
        selectTokenIndex = np.searchsorted(sumOfProbs, randomFloat) # find the index of the token in redTokenSet that corresponds to the random float
        if selectTokenIndex >= len(redTokenSet):
            selectTokenIndex = len(redTokenSet) - 1
        
        selectedToken = redTokenSet[selectTokenIndex]
        
        return selectedToken, selectTokenIndex, sumOfProbs[-1]
         

    
    def initiateWallet(self):
        """
        Initialize the wallet with a set of tokens.
        """
        self.highThroughputWallet = Wallet()
        i = self.globalTokenIndex
        
        currentTransactionValue = self.transactionSet[self.currentTransactionIndex]
        
        if currentTransactionValue <= 0.0:
            print("Warning: current transaction value is zero or negative, no tokens can be selected for initialization.")
            return
        
        
        while currentTransactionValue > 0.0:
            val = self.coinSelectionDistr.pickValueFromContinuousDistribution(currentTransactionValue)
            token = Token(val, serialno=self.globalTokenIndex)
            self.highThroughputWallet.addToken(token)
            self.globalTokenIndex += 1
            currentTransactionValue -= val
        
        
        if 
        
        self.currentTransactionIndex += 1
        
        
        
        
        
        
        
    
    def setTransactionsForSimulation(self, transactions):
        """
        Set the transactions for the simulation.
        """
        self.transactionSet = transactions
        self.transactionSetSize = len(self.transactionSet)
        self.coinCountPerTransaction = [0] * self.transactionSetSize
        self.currentTransactionIndex = 0
        
    def prolongTransactionSet(self, newTransactions):
        """
        Prolong the transaction set with new transactions.
        """
        self.transactionSet.extend(newTransactions)
        self.transactionSetSize = len(self.transactionSet)
        self.coinCountPerTransaction.extend([0] * len(newTransactions))
        