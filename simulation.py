# Created by Marc Winstel on July 17, 2025
import numpy as np
from transaction import initializeRandomNumGenerator, generateUniformFloats 
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet


class SimulationHandler:
    """
    Class to handle simulation of coinselection.
    """
    def __init__(self, tokenDenominationBuckets, beta = 0.1, drawDepositToken = False, adjustBetaAfterEachTransaction = False, doEmergRefund = True, useBucketsForProbabilityComp = False,mode="canonical"):
        self.useBucketsForProbabilityComp = useBucketsForProbabilityComp ## in case this is true, one computes only the probabilities for the average value of the bucket. After a bucket is selected, one draws a random token from within the bucket
        ### for transaction handling
        self.transactionSet = [1e07] # Initialize with a single transaction, 2e03 for dirichlet, 1e05 for normal distibution
        self.depositMode = "singletoken" # "singletoken" or "drawtokenFlexibleBeta"
        self.adjustBetaAfterEachTransaction = adjustBetaAfterEachTransaction # triggers self.adjustBetaMicrocanonically() after each transaction
        self.distMode = mode # "canonical", "grandcanonical", "uniform"
        self.doEmergenceRefund = doEmergRefund # If True, the emergence refund is triggered if the total value of the wallet is below a certain threshold
        if self.distMode == "uniform":
            self.adjustBetaAfterEachTransaction = False # uniform mode does not use beta, so we do not adjust it after each transaction
        if self.distMode not in ["canonical", "grandcanonical", "uniform"]:
            raise ValueError("Invalid mode. Choose from 'canonical', 'grandcanonical', or 'uniform'.")
        
        self.eps = 1e-05   # lower absolute value threshold to ignore small values in the simulation

        self.currentTransactionIndex = 0
        self.transactionSetSize = len(self.transactionSet)
        self.tokenCountInvolvedInTransaction = [None] * self.transactionSetSize # used to store the number of tokens involved in each transaction
        self.totalValueHistory = [None] * self.transactionSetSize # used to store the total value of the wallet after each transaction
        self.tokenCountHistory = [None] * self.transactionSetSize # used to store the number of tokens in the wallet after each transaction
        self.saveBetaHistory = [None] * self.transactionSetSize# used to store the beta value after each transaction
        self.emergenceRefundTransactionIndices = [] # used to store the indices of transactions where the emergence refund was triggered

        self.__globalTokenIndex = 0 # used to assign unique serial numbers to tokens!!!! Pay attention when changing this 

        self.ownrng = initializeRandomNumGenerator()


        if drawDepositToken:
            self.depositMode = "drawtokenFlexibleBeta"
        
        self.tokenBuckets = tokenDenominationBuckets # This is a list of token denomination buckets, e.g. [1, 1e01, 1e02] or [2e-01, 2e00, 2e01, 2e02] 
        self.tokenNoPerBucket = [0] * len(self.tokenBuckets) # This is a list of the number of tokens in each bucket, initialized to zero
        self.highThroughputWallet = Wallet()
        self.coinSelectionDistr = CoinSelectionDistribution(beta=beta, tokenDenominationBuckets=tokenDenominationBuckets, distMode=mode) # Initialize the coin selection distribution with the given beta and token denomination buckets
        
        
        self.initiateWallet()  # Initialize the wallet with a set of tokens

        self.timeCounter = 0 ## used for tracking special cases in the simulation, see self.handleNextTransaction()

        if self.coinSelectionDistr.mode == "grandcanonical":
            print("Warning: CoinSelectionDistribution is in grandcanonical mode, this is not tested very well.")
            print("Currently mu is by defualt set to 0.0")
    
    def simulateCurrentTransactionSet(self):
        """
        Simulate the current transaction set.
        """
        while self.currentTransactionIndex < self.transactionSetSize:
            self.handleNextTransaction()
        
        print("Simulation completed.")

    def selectTokenFromDistribtion(self, transactionValue):
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

    def selectTokenBucketFromDistributionThenPickRandom(self):   
        """
        Select a token bucket from the distribution, then pick a random token from that bucket.
        """
        probs, intBoundsForDrawing = self.coinSelectionDistr.returnBucketProbabilitiesForFixedWalletState(tokenNoPerBucket=self.tokenNoPerBucket)
        #print("probs = ", probs)
        #print("intBoundsForDrawing = ", intBoundsForDrawing)
        randomUniform = generateUniformFloats(self.ownrng, 1, 0.0, intBoundsForDrawing[-1]) # generate a random float in the range [0, intBoundsForDrawing[-1])
        selectBucketIndex = np.searchsorted(intBoundsForDrawing, randomUniform, side='right')[0] #

        tokenInBucket = self.findAllTokensInCertainBucket(selectBucketIndex) # find all tokens in the selected bucket

        if tokenInBucket == []:
            print("Warning: No tokens found in the selected bucket. Returning [].")
            print("probs = ", probs)
            print("intBoundsForDrawing = ", intBoundsForDrawing)
            print("selectBucketIndex = ", selectBucketIndex, "intBoundsForDrawing = ", intBoundsForDrawing)
            print("Wallet state:", self.highThroughputWallet)
            print("Token buckets:", self.tokenBuckets)
            print("Token no per bucket:", self.tokenNoPerBucket )
            return [], -1, intBoundsForDrawing[-1]
        
        selectToken = self.ownrng.choice(tokenInBucket) # randomly select a token from the bucket
        #print("Selected token from bucket:", selectToken, "from bucket index:", selectBucketIndex, "with value:", selectToken.value)
        return selectToken, selectBucketIndex, intBoundsForDrawing[-1]
    
    def tokenSelectionProcess(self, transactionValue):
        """
        Process the token selection. Currently, the transactionValue is not used, but it might be used in other algorithms to select tokens based on the transaction value.
        """
        if self.useBucketsForProbabilityComp:
            selectedToken, selectBucketIndex, sumOfBucketProbs = self.selectTokenBucketFromDistributionThenPickRandom()
            if selectedToken is None:
                print("Warning: No suitable token found in the selected bucket.")
                return None, -1
            return selectedToken, selectBucketIndex, sumOfBucketProbs
        else:
            selectedToken, selectTokenIndex, sumOfProbs = self.selectTokenFromDistribtion(transactionValue)
            if selectedToken == []:
                print("Warning: No suitable token found for the transaction value.")
                return None, -1
            return selectedToken, selectTokenIndex, sumOfProbs

    def findAllTokensInCertainBucket(self, bucketIndex):
        """
        Find all tokens in a certain bucket.
        """
        if bucketIndex < 0 or bucketIndex >= len(self.tokenBuckets):
            print("Warning: Bucket index out of range.")
            return []
        
        tokensInBucket = [token for token in self.highThroughputWallet.tokens if self.getTokensTokenBuckets(token.value) == bucketIndex]
        return tokensInBucket


    def addTokenToOwnWallet(self, token):
        """
        Add a token to the wallet. This increments the global token index in the simulation.
        """
        self.highThroughputWallet.addToken(token)
        self.__globalTokenIndex += 1
        # Add +1 to the tokenNoPerBucket for the bucket the token belongs to
        bucketIndex = self.getTokensTokenBuckets(token.value)
        if bucketIndex != -1:
            self.tokenNoPerBucket[bucketIndex] += 1
        else:
            print("Warning: Token value", token.value, "does not fit into any bucket. Please check the token denomination buckets.")
        #print("Added token with value", token.value, "and serial number", token.sno, "to the wallet. Current global token index is", self.__globalTokenIndex)
        #print("Curren tokenNoPerBucket is", self.tokenNoPerBucket)
           
    def removeTokenOwnWallet(self, token):
        """
        Remove a token from the wallet by its value.
        """
        if self.highThroughputWallet.searchTokenBySno(token.sno) is None:
            print("Warning: Token with serial number", token.sno, "not found in the wallet.")
            return
        else:
            self.highThroughputWallet.removeTokenBySno(token.sno)
            bucketIndex = self.getTokensTokenBuckets(token.value)
            if bucketIndex != -1:
                self.tokenNoPerBucket[bucketIndex] -= 1
            else:
                print("Warning: Token value", token.value, "does not fit into any bucket. Please check the token denomination buckets.")



    def adjustBetaMicrocanonically(self):
        if self.depositMode == "uniform":
            self.coinSelectionDistr.setBeta(0.0)
            if self.coinSelectionDistr.mode != "uniform":
                self.coinSelectionDistr.setMode("uniform")
        else:
            totalValue = self.highThroughputWallet.getTotalValue()
            if totalValue == 0.0:
                totalValue = 1.0  # Avoid division by zero
            tokenCount = self.highThroughputWallet.getTokenCount()
            if tokenCount == 0:
                tokenCount = 1
            self.coinSelectionDistr.setBeta(tokenCount / totalValue)  # Adjust beta based on the number of tokens and total value in the wallet
    
    def adjustMuArrayBucketWise(self):
        """
        Adjust the mu array bucket-wise based on the current wallet state.
        This is used in grandcanonical mode to adjust the chemical potential.
        """
        if self.coinSelectionDistr.mode != "grandcanonical":
            print("Warning: CoinSelectionDistribution is not in grandcanonical mode, not meaningful to adjust mu array.")
            return
        
        totalValue = self.highThroughputWallet.getTotalValue()
        if totalValue == 0.0:
            totalValue = 1.0
        tokenCount = self.highThroughputWallet.getTokenCount()
        if tokenCount == 0:
            tokenCount = 1
        beta = self.coinSelectionDistr.beta
        noBuckets = len(self.tokenBuckets)
        desiredNoPerBucket = tokenCount / noBuckets
        muValueArray = [0.0] * noBuckets
        for i in range(noBuckets):
            muValue = None
            if i != 0:
                muValue = (self.tokenBuckets[i-1] + self.tokenBuckets[i]) / 2.0 + np.log(desiredNoPerBucket) / beta
            else:
                muValue = self.tokenBuckets[i] / 2.0 + np.log(desiredNoPerBucket) / beta
            muValueArray[i] = muValue
        self.coinSelectionDistr.setMuArray(muValueArray)

    def initiateWallet(self):
        """
        Initialize the wallet with a set of tokens.
        """
        self.highThroughputWallet = Wallet()
        i = self.__globalTokenIndex
        
        currentTransactionValue = self.transactionSet[self.currentTransactionIndex]
        
        if currentTransactionValue < -self.eps:
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

        
    def getTokensTokenBuckets(self, valueOfToken):
        """
        Get the token buckets for a given token value.
        This is used to find the appropriate bucket for a token based on its value.
        """
        for i, bucket in enumerate(self.tokenBuckets):
            if valueOfToken < bucket and (i == 0 or valueOfToken >= self.tokenBuckets[i-1]):
                return i
        return -1

    
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
        
        if np.abs(depositValue) < self.eps:
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
                removeToken = self.highThroughputWallet.selectTokenBySno(self.__globalTokenIndex - 1) # select the last token added to the wallet
                val = removeToken.value # it must be the last token added to the wallet which led to negative depositValue
                depositValue += val
                self.removeTokenOwnWallet(removeToken) # remove the last token added to the wallet
                depositWallet.removeTokenBySno(removeToken.sno) # remove the last token added to the wallet
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
        
        while remainingPaymentValue > 0.0 and np.abs(remainingPaymentValue) > 1e-06:
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
            selectedToken, selectTokenIndex, sumOfProbs = self.tokenSelectionProcess(remainingPaymentValue)
            if selectedToken == []:
                print("No suitable token found for payment.")
                if self.highThroughputWallet.isEmpty():
                    print("Wallet is empty, cannot proceed with payment.")
                    raise ValueError("Wallet is empty, cannot proceed with payment.")
                    return
                else:
                    selectRandom = self.highThroughputWallet.selectTokenRandomly()
                    selectedToken = selectRandom
                    selectTokenIndex = selectRandom.sno 
            selectedWallet.addToken(selectedToken)
            self.removeTokenOwnWallet(selectedToken)  # Remove the token from the wallet, and adjust counters
            prevPaymentValue = remainingPaymentValue
            remainingPaymentValue -= selectedToken.value
            if self.adjustBetaAfterEachTransaction:
                self.adjustBetaMicrocanonically()

        if remainingPaymentValue > 0.0 and np.abs(remainingPaymentValue) > self.eps:
            print(f"Warning: Remaining payment value {remainingPaymentValue} could not be covered by available tokens.")
        if remainingPaymentValue < 0.0 and np.abs(remainingPaymentValue) > self.eps:
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
        if np.abs(currentTransactionValue) < self.eps:
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
        # if self.highThroughputWallet.getTokenCount() < 2 and self.currentTransactionIndex > 15: ## adjust this to whatever special behavior you want to be warned about
        #     if self.timeCounter > 5000:
        #         print("Warning: Maximal token value in wallet is less than 10^4, this might lead to unexpected behavior.")
        #         print("Maximal token value in wallet is", self.trackMaximalTokenValue)
        #         print("Current transaction index is", self.currentTransactionIndex)
        #         print("Current wallet tokens are", self.highThroughputWallet.tokens)
        #     self.timeCounter += 1

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
        if self.totalValueHistory[self.currentTransactionIndex] < 0.1*self.transactionSet[0] and self.doEmergenceRefund:
            print("Warning: Total wallet value is low.")
            self.emergenceRefund()  # Handle the emergence refund process if the total value is too low


        #self.trackMaximalTokenValue = max([t.value for t in self.highThroughputWallet.tokens])


    def emergenceRefund(self):
        """
        Handle the emergence refund process.
        """
        if not self.doEmergenceRefund:
            print("Emergence refund is not enabled, skipping.")
            return

        refundToken = Token(self.transactionSet[0], serialno=self.__globalTokenIndex)
        self.addTokenToOwnWallet(refundToken)
        print("Emergence refund triggered. Adding token with value", refundToken.value, "to the wallet. Current transaction index is", self.currentTransactionIndex)
        self.emergenceRefundTransactionIndices.append(self.currentTransactionIndex)
