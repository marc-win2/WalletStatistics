# Created by Marc Winstel on July 17, 2025
import numpy as np
from transaction import initializeRandomNumGenerator, generateUniformFloats 
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet


class SimulationHandler:
    """
    Class to handle simulation of coinselection.
    """
    def __init__(self, tokenDenominationBuckets, beta = 0.1, drawDepositToken = False, adjustBetaAfterEachTransaction = False, mode="canonical"):
        ### for transaction handling
        self.transactionSet = [1e07] # Initialize with a single transaction       
        self.depositMode = "singletoken" # "singletoken" or "drawtokenFlexibleBeta"
        self.adjustBetaAfterEachTransaction = adjustBetaAfterEachTransaction # triggers self.adjustBetaMicrocanonically() after each transaction
        self.distMode = mode # "canonical", "grandcanonical", "uniform"
        if self.distMode == "uniform":
            self.adjustBetaAfterEachTransaction = False # uniform mode does not use beta, so we do not adjust it after each transaction
        if self.distMode not in ["canonical", "grandcanonical", "uniform"]:
            raise ValueError("Invalid mode. Choose from 'canonical', 'grandcanonical', or 'uniform'.")

        self.currentTransactionIndex = 0
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountInvolvedInTransaction = [None] * self.transactionSetSize # used to store the number of tokens involved in each transaction
        self.totalValueHistory = [None] * self.transactionSetSize # used to store the total value of the wallet after each transaction
        self.tokenCountHistory = [None] * self.transactionSetSize # used to store the number of tokens in the wallet after each transaction
        self.saveBetaHistory = [None] * self.transactionSetSize# used to store the beta value after each transaction


        self.__globalTokenIndex = 0 # used to assign unique serial numbers to tokens!!!! Pay attention when changing this 

        self.ownrng = initializeRandomNumGenerator()


        if drawDepositToken:
            self.depositMode = "drawtokenFlexibleBeta"
        
    
        self.highThroughputWallet = Wallet()
        self.coinSelectionDistr = CoinSelectionDistribution(beta=beta, tokenDenominationBuckets=tokenDenominationBuckets, distMode=mode) # Initialize the coin selection distribution with the given beta and token denomination buckets
        
        
        self.initiateWallet()  # Initialize the wallet with a set of tokens

        self.timeCounter = 0 ## used for tracking special cases in the simulation, see self.handleNextTransaction()

        if self.coinSelectionDistr.mode == "grandcanonical":
            print("Warning: CoinSelectionDistribution is in grandcanonical mode, this is not tested very well.")
    
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

        probs, sumOfProbs = self.coinSelectionDistr.compDistributionDiscrSet(self.highThroughputWallet.tokens, transactionValue) # redTokenSet is the reduced token set with values <= transactionValue, sumOfProbs[i] contains the sum of probabilities for all tokens up to index i in redTokenSet, probs[i] contains the probability for the token at index i in redTokenSet
        if sumOfProbs == []:
            print("Warning: sumOfProbs is empty, returning None.")
            print("    tokenSet = ", self.highThroughputWallet.tokens, "transactionValue = ", transactionValue, "probs = ", probs, "sumOfProbs = ", sumOfProbs)

        randomFloat = generateUniformFloats(self.ownrng, 1, 0.0, sumOfProbs[-1]) # generate a random float in the range [0, sumOfProbs[-1]) where sumOfProbs[-1] is the sum of all probabilities
        
        selectTokenIndex = np.searchsorted(sumOfProbs, randomFloat)[0] # find the index of the token in redTokenSet that corresponds to the random float

        
        selectedToken = self.highThroughputWallet.tokens[selectTokenIndex] if selectTokenIndex < len(self.highThroughputWallet.tokens) else [] # select the token from the wallet based on the index
        
        return selectedToken, selectTokenIndex, sumOfProbs[-1]
         

    def addTokenToOwnWallet(self, token):
        """
        Add a token to the wallet. This increments the global token index in the simulation.
        """
        self.highThroughputWallet.addToken(token)
        self.__globalTokenIndex += 1

    def adjustBetaMicrocanonically(self):
        if self.depositMode == "uniform":
            self.coinSelectionDistr.setBeta(0.0)
            self.coinSelectionDistr.setMode("uniform")
            return
        totalValue = self.highThroughputWallet.getTotalValue()
        if totalValue == 0.0:
            totalValue = 1.0  # Avoid division by zero
        tokenCount = self.highThroughputWallet.getTokenCount()
        if tokenCount == 0:
            tokenCount = 1
        self.coinSelectionDistr.setBeta(tokenCount / totalValue)  # Adjust beta based on the number of tokens and total value in the wallet
        

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

        self.tokenCountInvolvedInTransaction[self.currentTransactionIndex] = initiationWallet.getTokenCount()
        if self.adjustBetaAfterEachTransaction:
            self.adjustBetaMicrocanonically()

        self.doWalletStateTracking()  # Track the number of tokens in the wallet after each transaction and the total value of the wallet


        self.currentTransactionIndex += 1

        
            
    
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
                if self.adjustBetaAfterEachTransaction:
                    self.adjustBetaMicrocanonically()

        
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
        
        freezeOverallWallet = self.highThroughputWallet.tokens
        remainingPaymentValue = -paymentValue
        selectedWallet = Wallet()
        
        while remainingPaymentValue > 0.0:
            if self.highThroughputWallet.isEmpty():
                print("Wallet is empty, cannot proceed with payment.")
                print("Overall payment value was", -paymentValue)
                print("Previous payment value was ", prevPaymentValue)
                print("Selected tokens are", selectedWallet)
                print("Their sum is", selectedWallet.getTotalValue())
                print("Remaining payment value is", remainingPaymentValue   )
                print("Previous selected token was", selectedToken  )
                print("Old wallet tokens were", freezeOverallWallet)
                print("current transaction index is", self.currentTransactionIndex)
            selectedToken, selectTokenIndex, sumOfProbs = self.selectTokenFromDistribtion(remainingPaymentValue)
            if selectedToken == []:
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
            prevPaymentValue = remainingPaymentValue
            remainingPaymentValue -= selectedToken.value
            if self.adjustBetaAfterEachTransaction:
                self.adjustBetaMicrocanonically()

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
        
        trueTransaction = True
        if np.abs(currentTransactionValue) < 1e-06:
            self.tokenCountInvolvedInTransaction[self.currentTransactionIndex] = 0
            trueTransaction = False
            # Skip if the transaction value is effectively zero

        newTokens = None
        if currentTransactionValue < 0.0 and trueTransaction: ### payment
            removedTokens = self.handlePayment(paymentValue=currentTransactionValue)
            self.tokenCountInvolvedInTransaction[self.currentTransactionIndex] = removedTokens.getTokenCount()
            self.totalValueHistory[self.currentTransactionIndex] = removedTokens.getTotalValue()

        if currentTransactionValue > 0.0 and trueTransaction: ## deposit
            newTokens = self.handleDeposit(depositValue=currentTransactionValue)
            self.tokenCountInvolvedInTransaction[self.currentTransactionIndex] = newTokens.getTokenCount()
        
        if self.adjustBetaAfterEachTransaction:
            self.adjustBetaMicrocanonically()
        
        self.doWalletStateTracking()  # Track the number of tokens in the wallet after each transaction and the total value of the wallet
        self.trackMaximalTokenValue = max([t.value for t in self.highThroughputWallet.tokens])
        if self.highThroughputWallet.getTokenCount() < 2 and self.currentTransactionIndex > 15: ## adjust this to whatever special behavior you want to be warned about
            if self.timeCounter > 5000:
                print("Warning: Maximal token value in wallet is less than 10^4, this might lead to unexpected behavior.")
                print("Maximal token value in wallet is", self.trackMaximalTokenValue)
                print("Current transaction index is", self.currentTransactionIndex)
                print("Current wallet tokens are", self.highThroughputWallet.tokens)
            self.timeCounter += 1

        self.currentTransactionIndex += 1


        
    
        
    def prolongTransactionSet(self, newTransactions):
        """
        Prolong the transaction set with new transactions. Typically used because the transaction set is initialized with a single transaction.
        """
        self.transactionSet.extend(newTransactions)
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountInvolvedInTransaction.extend([None] * len(newTransactions))
        self.totalValueHistory.extend([None] * len(newTransactions))
        self.tokenCountHistory.extend([None] * len(newTransactions))
        self.saveBetaHistory.extend([None] * len(newTransactions))

    def doWalletStateTracking(self):
        """
        Track the number of tokens in the wallet after each transaction and the total value of the wallet.
        """
        if self.currentTransactionIndex >= self.transactionSetSize:
            print("No more transactions to track.")
            return
        self.saveBetaHistory[self.currentTransactionIndex] = self.coinSelectionDistr.beta
        self.tokenCountHistory[self.currentTransactionIndex] = self.highThroughputWallet.getTokenCount()
        self.totalValueHistory[self.currentTransactionIndex] = self.highThroughputWallet.getTotalValue()
        self.trackMaximalTokenValue = max([t.value for t in self.highThroughputWallet.tokens])
