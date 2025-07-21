# Created by Marc Winstel on July 17, 2025
import numpy as np
from transaction import initializeRandomNumGenerator, generateUniformFloats 
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet


class SimulationHandler:
    """
    Class to handle simulation of coinselection.
    """
    def __init__(self, tokenDenominationBuckets, beta = 0.1, drawDepositToken = False, adjustBetaAfterEachTransaction = False):
        ### for transaction handling
        self.transactionSet = [100000.0]      
        self.depositMode = "singletoken" # "singletoken" or "drawtokenFlexibleBeta"
        self.adjustBetaAfterEachTransaction = adjustBetaAfterEachTransaction # triggers self.adjustBetaMicrocanonically() after each transaction


        self.currentTransactionIndex = 0
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountPerTransaction = [None] * self.transactionSetSize
        self.__globalTokenIndex = 0 # used to assign unique serial numbers to tokens!!!! Pay attention when changing this 



        if drawDepositToken:
            self.depositMode = "drawtokenFlexibleBeta"
        
    
        self.highThroughputWallet = Wallet()
        self.coinSelectionDistr = CoinSelectionDistribution(beta, tokenDenominationBuckets)
        self.coinSelectionDistr.setCanonical()
        
        
        self.initiateWallet()  # Initialize the wallet with a set of tokens
    
    def simulateCurrentTransactionSet(self):
        """
        Simulate the current transaction set.
        """
        while self.currentTransactionIndex < self.transactionSetSize:
            self.handleNextTransaction()
        
        print("Simulation completed.")

    def selectTokenFromDistribtion(self,transactionValue):
        """
        Select a token from the distribution based on the transaction value.
        """
        probs, sumOfProbs, redTokenSet = self.coinSelectionDistr.compDistributionDiscrSet(self.highThroughputWallet.tokens, transactionValue) # redTokenSet is the reduced token set with values <= transactionValue, sumOfProbs[i] contains the sum of probabilities for all tokens up to index i in redTokenSet, probs[i] contains the probability for the token at index i in redTokenSet
                
        rng = initializeRandomNumGenerator()
        randomFloat = generateUniformFloats(rng, 1, 0.0, sumOfProbs[-1]) # generate a random float in the range [0, sumOfProbs[-1]) where sumOfProbs[-1] is the sum of all probabilities
        
        selectTokenIndex = np.searchsorted(sumOfProbs, randomFloat)[0] # find the index of the token in redTokenSet that corresponds to the random float
        if selectTokenIndex >= len(redTokenSet):
            selectTokenIndex = len(redTokenSet) - 1
        
        selectedToken = redTokenSet[selectTokenIndex]
        
        return selectedToken, selectTokenIndex, sumOfProbs[-1]
         

    def addTokenToOwnWallet(self, token):
        """
        Add a token to the wallet. This increments the global token index in the simulation.
        """
        self.highThroughputWallet.addToken(token)
        self.__globalTokenIndex += 1

    def adjustBetaMicrocanonically(self):
        self.beta = self.highThroughputWallet.getTokenCount() / self.highThroughputWallet.getTotalValue()
    
    def initiateWallet(self):
        """
        Initialize the wallet with a set of tokens.
        """
        self.highThroughputWallet = Wallet()
        i = self.__globalTokenIndex
        
        currentTransactionValue = self.transactionSet[self.currentTransactionIndex]
        
        if currentTransactionValue <= 0.0:
            print("Warning: current transaction value is zero or negative, no tokens can be selected for initialization.")
            raise ValueError("Cannot initialize wallet with zero or negative transaction value.")

        if self.currentTransactionIndex != 0:
            print("Warning: currentTransactionIndex is not zero, this might lead to unexpected behavior.")
        
        if self.highThroughputWallet.isEmpty() == False:
            print("Warning: Wallet is not empty, this might lead to unexpected behavior.")

        initiationWallet = self.handleDeposit(depositValue=currentTransactionValue)

        self.tokenCountPerTransaction[self.currentTransactionIndex] = initiationWallet.getTokenCount()

        self.currentTransactionIndex += 1

        if self.adjustBetaAfterEachTransaction:
            self.adjustBetaMicrocanonically
            
    
    def handleDeposit(self, depositValue):
        """
        Handle a deposit transaction by adding tokens to the wallet.
        Creates new tokens for the whole system and adds them to the wallet.
        New serial numbers are assigned to the tokens.
        the globalTokenIndex is incremented for each new token added.
        """
        if depositValue < 0.0:
            print("Deposit value must be positive.")
            return Wallet()
        
        if np.abs(depositValue) < 1e-06:
            print("Deposit value is effectively zero, no tokens will be added.")
            return Wallet()
        
        depositWallet = Wallet()

        originalDepositValue = depositValue 

        if self.depositMode == "singletoken":
            # Create a single token with the full deposit value
            token = Token(depositValue, serialno=self.__globalTokenIndex)
            self.addTokenToOwnWallet(token)
            depositWallet.addToken(token)

        elif self.depositMode == "drawtokenFlexibleBeta":
            while depositValue > 0.0:
                val = self.coinSelectionDistr.pickValueFromContinuousDistributionWithBetaAdjustment(originalDepositValue)
                token = Token(val, serialno=self.__globalTokenIndex)
                self.addTokenToOwnWallet(token) # increment globalTokenIndex because a new token is added and sno are uniquely assigned
                depositWallet.addToken(token)
                depositValue -= val

        
            if depositValue < 0.0:
                val = self.highThroughputWallet.getTokenValue(self.__globalTokenIndex - 1) # it must be the last token added to the wallet which led to negative depositValue
                depositValue += val
                self.highThroughputWallet.removeTokenBySno(self.__globalTokenIndex - 1)
                depositWallet.removeTokenBySno(self.__globalTokenIndex - 1)
                newToken = Token(depositValue, serialno=self.__globalTokenIndex - 1)
                self.highThroughputWallet.addToken(newToken) # use the walletmemberfunction to add the token such that the globalTokenIndex is not incremented again
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
        selectedWallet = Wallet()
        
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
            selectedWallet.addToken(selectedToken)
            self.highThroughputWallet.removeTokenBySno(selectedToken.sno)
            remainingPaymentValue -= selectedToken.value
            
        
        if remainingPaymentValue > 0.0:
            print(f"Warning: Remaining payment value {remainingPaymentValue} could not be covered by available tokens.")
        if remainingPaymentValue < 0.0:
            token = Token(-remainingPaymentValue, serialno=self.__globalTokenIndex)
            self.addTokenToOwnWallet(token)  # Add the remaining value as a new token
            selectedWallet.addToken(token)
        
        return selectedWallet


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

        newTokens = None
        if currentTransactionValue < 0.0: ### payment
            removedTokens = self.handlePayment(paymentValue=currentTransactionValue)
            self.tokenCountPerTransaction[self.currentTransactionIndex] = removedTokens.getTokenCount()
            
        
        if currentTransactionValue > 0.0: ## deposit
            newTokens = self.handleDeposit(depositValue=currentTransactionValue)
            self.tokenCountPerTransaction[self.currentTransactionIndex] = newTokens.getTokenCount()
        
        if self.adjustBetaAfterEachTransaction:
            self.adjustBetaMicrocanonically()

        self.currentTransactionIndex += 1

        
        
        
        
        
        
        
    
    def setTransactionsForSimulation(self, transactions):
        """
        Set the transactions for the simulation.
        Currently not used, because wallet is typically initialized with a single transaction, which can then be prolonged.
        """
        self.transactionSet = transactions
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountPerTransaction = [None] * self.transactionSetSize
        self.currentTransactionIndex = 0
        
    def prolongTransactionSet(self, newTransactions):
        """
        Prolong the transaction set with new transactions. Typically used because the transaction set is initialized with a single transaction.
        """
        self.transactionSet.extend(newTransactions)
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountPerTransaction.extend([None] * len(newTransactions))
        