# Created by Marc Winstel on 14.07.25
import numpy as np 
import matplotlib.pyplot as plt
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

def coinSelectionDistributionTest():
    coinDistrTest = CoinSelectionDistribution(1e-02, tokenDenominationBuckets)
    #coinDistrTest.setCanonical()
    print("coinDistr.muArray = ", coinDistrTest.betaMuArray)
    print("coinDistr.expectedTokenNoPerBucket = ", coinDistrTest.expn)

def simulationTest(tokenDenominationBuckers):
    simulate = SimulationHandler(tokenDenominationBuckets, 1e-03, drawDepositToken=False)
    simulate.coinSelectionDistr.setCanonical()
    print("SimulationHandler initialized.")
    print("simulate.highThroughputWallet = ", simulate.highThroughputWallet.getTokenCount())
    print("simulate.highThroughputWallet = ", simulate.highThroughputWallet)    
    print("simulate.highThroughputWallet.getTotalValue() = ", simulate.highThroughputWallet.getTotalValue())

    simulate.prolongTransactionSet(transactions)
    print("simulate.transactionSetSize = ", simulate.transactionSetSize)
    print("simulate.tokenCountPerTransaction[0] = ", simulate.tokenCountPerTransaction[0])

    print("now simulate 5 transactions, namely", simulate.transactionSet[1:6])

    simulate.handleNextTransaction()    
    print(simulate.highThroughputWallet)

    simulate.handleNextTransaction()    
    print(simulate.highThroughputWallet)

    simulate.handleNextTransaction()       
    print(simulate.highThroughputWallet)

    simulate.handleNextTransaction()   
    print(simulate.highThroughputWallet)

    simulate.handleNextTransaction()    
    print(simulate.highThroughputWallet)



if __name__ == "__main__":
    tokens =  [2**i for i in range(0, 30)]
    tokenDenominationBuckets = tokens# np.append([0], tokens)
    #print("tokenDenominationBuckets = ", tokenDenominationBuckets)
    
    #plottingTransactionsTest()   
    
    #coinSelectionDistributionTest()

    transactionGenerator = RandomTransactionGenerator()
    transactionGenerator.maxAbsTransactionValue = 10**7### maximal, absolute transaction value
     # signature of generateNTransactionsGaussian(n, stdDev, mean), transactions can in principle be 
     # negative and positive, corresponding to deposits and withdrawals
    xFactor = 3
    noPayments = 1000
    deposits = transactionGenerator.generateNTransactionsGaussian(xFactor*noPayments, 250, 1000)
    print("deposits= " , np.mean(deposits), "+-" ,np.std(deposits))
    payments = transactionGenerator.generateNTransactionsGaussian(noPayments, 500, -3000)
    print("payments= " , np.mean(payments), "+-" ,np.std(payments))


    # Merge deposits and payments so that one payment follows three deposits
    transactions = []
    deposit_idx = 0
    payment_idx = 0
    while deposit_idx + 3 <= len(deposits) and payment_idx < len(payments):
        transactions.append(payments[payment_idx])

        # Add xFactor deposits
        transactions.extend(deposits[deposit_idx:deposit_idx+xFactor])
        deposit_idx += 3
        # Add one payment
        payment_idx += 1
    print("len(transactions) = ", len(transactions))

    plt.hist(deposits, bins=200, density=False, alpha=0.7, color='blue', label='Deposits')
    plt.hist(payments, bins=200, density=False, alpha=0.7, color='red', label='Payments')
    plt.title('Transactions Histogram')
    plt.xlabel('Value')
    plt.legend()
    plt.savefig('Data/transactions_histogram.png')
    with open('Data/transaction.dat', 'w') as f:
        for t in transactions:
            f.write(f"{t}\n")
    
            


 
    simulationTest(tokenDenominationBuckets)
    