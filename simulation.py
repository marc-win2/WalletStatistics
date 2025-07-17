# Created by Marc Winstel on July 17, 2025
import numpy as np

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
        
    
    
    def initiateWallet(self):
        """
        Initialize the wallet with a set of tokens.
        """
        self.highThroughputWallet = Wallet()
        i = self.globalTokenIndex
        currentTransactionValue = self.transactionSet[self.currentTransactionIndex]
        
        while currentTransactionValue > 0.0:
            a = 0 #val = self.coinSelectionDistr. 
        
        
        
        
        
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
        