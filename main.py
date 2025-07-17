# Created by Marc Winstel on 14.07.25
import numpy as np 
from scipy.integrate import quad
from transaction import RandomTransactionGenerator
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet
from simulation import SimulationHandler

def plottingTransactionsTest():
    transactionGenerator = RandomTransactionGenerator()
    
    transactionGenerator.maxAbsTransactionValue = 10**6### maximal, absolute transaction value
    
    
    #transactionGenerator.plotGaussianTransactionTestOne(10**6)
    
    transactions = transactionGenerator.plotGaussianWithUniformOutliersTransactionTestOne(10**5)
    print(transactions.mean(), transactions.std())


if __name__ == "__main__":
    tokens =  [2**i for i in range(0, 30)]
    tokenDenominationBuckets = tokens# np.append([0], tokens)
    print("tokenDenominationBuckets = ", tokenDenominationBuckets)
        
    coinDistr = CoinSelectionDistribution(0.1, tokenDenominationBuckets)
    print("coinDistr.muArray = ", coinDistr.muArray)
    print("coinDistr.expectedTokenNoPerBucket = ", coinDistr.expn)

    
    a = Token(2.0, serialno=0)
    print(a)
    b = Token(3.0, serialno=1)
    print(b)
    wallet = Wallet([a, b])
    print(wallet)
    
    
    