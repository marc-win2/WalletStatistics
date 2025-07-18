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
            raise ValueError("Cannot initialize wallet with zero or negative transaction value.")
            
        initiationWallet = self.handleDeposit(depositValue=currentTransactionValue)

        self.currentTransactionIndex += 1
    
    def handleDeposit(self, depositValue):
        """
        Handle a deposit transaction by adding tokens to the wallet.
        """
        if depositValue <= 0.0:
            print("Deposit value must be positive.")
            return
        
        depositWallet = Wallet()

        while depositValue > 0.0:
            val = self.coinSelectionDistr.pickValueFromContinuousDistribution(depositValue)
            token = Token(val, serialno=self.globalTokenIndex)
            self.highThroughputWallet.addToken(token)
            depositWallet.addToken(token)
            self.globalTokenIndex += 1
            depositValue -= val
        
        if depositValue < 0.0:
            val = self.highThroughputWallet.giveValue(self.globalTokenIndex - 1)
            depositValue += val
            self.highThroughputWallet.removeTokenBySno(self.globalTokenIndex - 1)
            newToken = Token(depositValue, serialno=self.globalTokenIndex - 1)
            self.highThroughputWallet.addToken(newToken)
            depositWallet.addToken(newToken)

        return depositWallet

    def handlePayment(self, paymentValue):
        """
        Handle a payment transaction by selecting tokens from the wallet.
        paymentValue should be negative, representing the amount to be paid.
        """
        if paymentValue >= 0.0:
            print("Payment value must be negative.")
            return
        
        remainingPaymentValue = -paymentValue
        selectedTokens = []
        
        while remainingPaymentValue > 0.0:
            selectedToken, selectTokenIndex, sumOfProbs = self.selectTokenFromDistribtion(remainingPaymentValue)
            if selectedToken is None:
                print("No suitable token smaller than bill value found for payment.")
                if self.highThroughputWallet.isEmpty():
                    print("Wallet is empty, cannot proceed with payment.")
                    raise ValueError("Wallet is empty, cannot proceed with payment.")
                    return
                else:
                    selectRandom = self.highThroughputWallet.selectTokenRandomly()
                    selectedToken = selectRandom
                    selectTokenIndex = selectRandom.sno 
            selectedTokens.append(selectedToken)
            remainingPaymentValue -= selectedToken.value
            
            # Remove the token from the wallet
            self.highThroughputWallet.removeTokenBySno(selectedToken.sno)
        
        if remainingPaymentValue > 0.0:
            print(f"Warning: Remaining payment value {remainingPaymentValue} could not be covered by available tokens.")
        
        return selectedTokens


    def handleNextTransaction(self):
        """
        Handle the next transaction by selecting tokens from the wallet.
        """
        if self.currentTransactionIndex >= self.transactionSetSize:
            print("No more transactions to handle.")
            return
        
        currentTransactionValue = self.transactionSet[self.currentTransactionIndex]
        
        if np.abs(currentTransactionValue) < 1e-06:
            self.currentTransactionIndex += 1    
            return# Skip if the transaction value is effectively zero


        if currentTransactionValue < 0.0: ### payment
            self.handlePayment(paymentValue=currentTransactionValue)
            self.currentTransactionIndex += 1
            return
            
        
        if currentTransactionValue > 0.0: ## deposit
            self.handleDeposit(depositValue=currentTransactionValue)
            self.currentTransactionIndex += 1
            return


        
        
        
        
        
        
        
    
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
        