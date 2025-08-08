# Created by Marc Winstel on 14.07.25
import numpy as np 
import matplotlib.pyplot as plt
import os
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

def simulationTest(tokenDenominationBuckers,  transactions):
    simulate = SimulationHandler(tokenDenominationBuckets, 1e-03, drawDepositToken=False)
    simulate.coinSelectionDistr.setCanonical()
    print("SimulationHandler initialized.")
    print("simulate.highThroughputWallet = ", simulate.highThroughputWallet.getTokenCount())
    print("simulate.highThroughputWallet = ", simulate.highThroughputWallet)    
    print("simulate.highThroughputWallet.getTotalValue() = ", simulate.highThroughputWallet.getTotalValue())

    simulate.prolongTransactionSet(transactions)
    print("simulate.transactionSetSize = ", simulate.transactionSetSize)
    print("simulate.tokenCountPerTransaction[0] = ", simulate.tokenCountInvolvedInTransaction[0])

    print("now simulate 9 transactions, namely", simulate.transactionSet[1:10])

    i = 0
    while i < 10:
        simulate.handleNextTransaction()
        print(simulate.highThroughputWallet)
        i += 1
    simulate.highThroughputWallet.removeTokenBySno(3)
    print("After removing token with serial number 3:")
    print(simulate.highThroughputWallet)




def plotTransactionData(transactions, payments, deposits, plottingIndex=0):
    plt.hist(deposits, bins=200, density=False, alpha=0.7, color='green', label='Deposits')
    plt.hist(payments, bins=200, density=False, alpha=0.7, color='red', label='Payments')
    plt.title('Transactions Histogram')
    plt.xlabel('Value')
    plt.legend()
    plt.savefig('Data/transactions_histogram' + str(plottingIndex) + '.png')
    plt.clf()  # Clear the current figure
    with open('Data/transaction' + str(plottingIndex) + '.dat', 'w') as f:
        for t in transactions:
            f.write(f"{t}\n")

def singleSimulation(transactions, tokenDenominationBuckets, simulationIndex, drawDeposit=False, adjustBeta=True, doEmergRefund=True, mode="canonical"):
    simulation = SimulationHandler(tokenDenominationBuckets=tokenDenominationBuckets, beta=1e-03, drawDepositToken=drawDeposit, adjustBetaAfterEachTransaction=adjustBeta, doEmergRefund=doEmergRefund, mode=mode, muGlobal=1e04)
    print("SimulationHandler initialized.")
    print("simulation.highThroughputWallet = ", simulation.highThroughputWallet)   
    simulation.prolongTransactionSet(transactions)
    print("first 3 transactions in the set:", simulation.transactionSet[:3])
    simulation.handleNextTransaction()  # Process the first transaction
    print("After processing first transaction:")
    print(simulation.highThroughputWallet)
    print("Total value in wallet:", simulation.highThroughputWallet.getTotalValue())

    ########### For testing purposes, uncomment the following lines to manually process more transactions
    #simulation.handleNextTransaction()  # Process the second transaction
    #print("After processing second transaction:")
    #print(simulation.highThroughputWallet)
    #print("Total value in wallet:", simulation.highThroughputWallet.getTotalValue())
    #simulation.handleNextTransaction()  # Process the third transaction
    #print("After processing third transaction:")
    #print(simulation.highThroughputWallet)
    #print("Total value in wallet:", simulation.highThroughputWallet.getTotalValue())
    #print("Token count in wallet:", simulation.highThroughputWallet.getTokenCount())


    simulation.simulateCurrentTransactionSet()
    print(simulation.highThroughputWallet)
    print(simulation.highThroughputWallet.getTotalValue())
    print(simulation.highThroughputWallet.getTokenCount())
    #print(simulation.tokenCountInvolvedInTransaction)
    print(np.mean(simulation.tokenCountInvolvedInTransaction))
    maxval = max(token.value for token in simulation.highThroughputWallet.tokens)
    print("Maximal token value in wallet:", maxval  )

    vals = [token.value for token in simulation.highThroughputWallet.tokens]

    vals.remove(maxval) 
    plt.hist(vals, bins=200,density=False)
    plt.title("Token values in Wallet after a single run")
    plt.xlabel("Token Value")
    plt.savefig('Data/token_values_histogram_' + str(simulationIndex) + '.png')
    with open('Data/token_values_' + str(simulationIndex) + '.dat', 'w') as f:
        for v in vals:
            f.write(f"{v}\n")
    
    transaction_ = np.arange(len(transactions)+1)
    with open('Data/WalletValue_' + str(simulationIndex) + '.dat', 'w') as f:
        for t, v in zip(transaction_, simulation.totalValueHistory):
            f.write(f"{t} {v}\n")
    plt.scatter(transaction_, simulation.totalValueHistory, marker='o', color='blue', linewidths=0.05)
    #plt.title('Total Value in Wallet per Transaction')
    plt.xlabel('Transaction Index')
    plt.ylabel('Total Value in Wallet')
    plt.savefig('Data/WalletValue_' + str(simulationIndex) + '.png')
    plt.clf()  # Clear the current figure

    with open('Data/TokenCount_' + str(simulationIndex) + '.dat', 'w') as f:
        for t, v in zip(transaction_, simulation.tokenCountHistory):
            f.write(f"{t} {v}\n")
    plt.scatter(transaction_, simulation.tokenCountHistory, marker='x', color='blue', linewidths=0.05)
    #plt.title('UTXO pool size')
    plt.xlabel('Transaction Index')
    plt.ylabel('UTXO Pool Size')
    plt.savefig('Data/TokenCount_' + str(simulationIndex) + '.png')
    plt.clf()  # Clear the current figure

    with open('Data/BetaPerTransaction_' + str(simulationIndex) + '.dat', 'w') as f:
        for t, b in zip(transaction_, simulation.saveBetaHistory):
            f.write(f"{t} {b}\n")
    plt.scatter(transaction_, simulation.saveBetaHistory, marker='o', color='black', linewidths=0.05)
    if np.any(simulation.saveBetaHistory) < 1e-10:
        plt.yscale('linear')
    else:
        plt.yscale('log')
    plt.title('Beta Value per Transaction')
    plt.xlabel('Transaction Index')
    plt.ylabel('Beta Value')
    plt.savefig('Data/BetaPerTransaction_' + str(simulationIndex) + '.png')
    plt.clf()  # Clear the current figure

    with open('Data/emergence_refund_indices_' + str(simulationIndex) + '.dat', 'w') as f:
        for idx in simulation.emergenceRefundTransactionIndices:
            f.write(f"{idx}\n")




    return vals, maxval, simulation.highThroughputWallet.getTotalValue(), simulation.highThroughputWallet.getTokenCount(), simulation.tokenCountInvolvedInTransaction, simulation.tokenCountHistory, simulation.totalValueHistory




def generateDoubleGaussianTransactionsAndPlotThem(plottingIndex=0,noPayments=100000, xFactor=3):
    transactionGenerator = RandomTransactionGenerator()
    transactionGenerator.maxAbsTransactionValue = 10**7### maximal, absolute transaction value
     # signature of generateNTransactionsGaussian(n, stdDev, mean), transactions can in principle be 
     # negative and positive, corresponding to deposits and withdrawals
    print("Generating Gaussian transactions... Payments and deposits separately" )
    deposits = []
    payments = []

    for i in range(noPayments):
        payments.append(transactionGenerator.generateTransactionGaussian(500, -3000))
        for b in range(xFactor):
            deposits.append(transactionGenerator.generateTransactionGaussian(250, 1000))
    print("deposits= " , np.mean(deposits), "+-" ,np.std(deposits))
    print("payments= " , np.mean(payments), "+-" ,np.std(payments))


    # Merge deposits and payments so that one payment follows three deposits
    transactions = []
    deposit_idx = 0
    payment_idx = 0
    while deposit_idx + xFactor <= len(deposits) and payment_idx < len(payments):
        transactions.append(payments[payment_idx])
        payment_idx += 1


        # Add xFactor deposits
        transactions.extend(deposits[deposit_idx:deposit_idx+xFactor])
        deposit_idx += xFactor

    print("len(transactions) = ", len(transactions))
    print("Mean of transactions = ", np.mean(transactions), "+-", np.std(transactions))

    plotTransactionData(transactions, payments, deposits, plottingIndex)

    
    return transactions, deposits, payments

def generateTransactions_PaymentsDirichlet_AndPlotThem(plottingIndex = 0, noDeposits = 100000, xFactor=10, generateDirichletAsFloats=True):

    print("Generating Dirichlet Payments and constant deposits")
    deposits = [2000] * noDeposits
    payments = []
    transactionGenerator = RandomTransactionGenerator()
    for i in range(noDeposits):
        if generateDirichletAsFloats:
            generateXPayments = transactionGenerator.generateTransactionDirichlet(1.0,sumValue=2000, sizealpha=xFactor)
        else:
            generateXPayments = transactionGenerator.generateIntegerDirichletPaymentsViaMultinomial(n=xFactor, sum=2000)
        for k in range(xFactor):
            payments.append(-1.0*generateXPayments[k])
    print("deposits= " , np.mean(deposits), "+-" ,np.std(deposits))
    print("payments= " , np.mean(payments), "+-" ,np.std(payments))

    # Merge deposits and payments so that one deposit follows xFactor payments
    transactions = []
    deposit_idx = 0
    payment_idx = 0
    while deposit_idx < len(deposits) and payment_idx + xFactor <= len(payments):
        # Add one deposit
        transactions.append(deposits[deposit_idx])
        deposit_idx += 1

        # Add xFactor payments
        transactions.extend(payments[payment_idx:payment_idx+xFactor])
        payment_idx += xFactor
    print("len(transactions) = ", len(transactions))
    print("Mean of transactions = ", np.mean(transactions), "+-", np.std(transactions))

    plotTransactionData(transactions, payments, deposits, plottingIndex)

    return transactions, deposits, payments


if __name__ == "__main__":
    tokens =  [10**i for i in range(-2, 10)]
    tokenDenominationBuckets = tokens# np.append([0], tokens)
    #print("tokenDenominationBuckets = ", tokenDenominationBuckets)
    
    #plottingTransactionsTest()   
    
    #coinSelectionDistributionTest()

    #transactions, deposits, payments = generateDoubleGaussianTransactionsAndPlotThem(plottingIndex=0, noPayments=100000, xFactor=3)
    #transactions, deposits, payments = generateTransactions_PaymentsDirichlet_AndPlotThem(plottingIndex=0, noDeposits=100000, xFactor=10)
    #simulationTest(tokenDenominationBuckets, transactions)


    if os.path.isdir('Data') is False:
        os.mkdir('Data')
    else:
        print("Data directory already exists, overwriting data files.")
        try:
            input("Press anything to continue or Ctrl+C to cancel...")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit(0)

    if os.path.isdir('DataGlobal') is False:
        os.mkdir('DataGlobal')
    else:
        print("DataGlobal directory already exists, appending to the data files.")
        print("Data Global files are always appended to, so the overall data might contain results from previous runs. Please check that this directory is empty before running the script.")
        try:
            input("Press anything to continue or Ctrl+C to cancel...")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit(0)

    # r = RandomTransactionGenerator()
    # safeInts = r.generateIntegerDirichletPaymentsViaMultinomial(n=10, sum=2000)
    # print("Generated Dirichlet payments via multinomial distribution:", safeInts)
    # print("Exp. value of Dirichlet payments:", np.mean(safeInts), "+-", np.std(safeInts))
    # safeFloats = r.generateTransactionDirichlet(alpha=1.0, sumValue=2000, sizealpha=10)  
    # print("Generated Dirichlet payments via Dirichlet distribution:", safeFloats)
    # print("Exp. value of Dirichlet payments:", np.mean(safeFloats), "+-", np.std(safeFloats))


    numSimulations = 10
    alltokenValues = []
    totalTransaction = []
    totalValues = []
    totalTokenCounts = []
    totalMaxTokenVals = []
    paymentTokenCountMeans = []
    paymentTokenCountStddev = []
    for i in range(numSimulations):
        transactions, deposits, payments = generateDoubleGaussianTransactionsAndPlotThem(plottingIndex=i, noPayments=100000, xFactor=3) # noDeposits = xFactor * noPayments
        #transactions, deposits, payments = generateTransactions_PaymentsDirichlet_AndPlotThem(plottingIndex=i, noDeposits=1000, xFactor=10) # noPayments = xFactor * noDeposits
        tokenVals, maxTokenValRemoved, totalValue, totalTokensInWallet, tokenCountPerT, tokenCountHistory, totalValueHistory = singleSimulation(transactions, tokenDenominationBuckets, i, drawDeposit=False, adjustBeta=False, doEmergRefund=True,  mode="grandcanonical")

        ## handle paymentTOkenCount here and generate Data and Plot
        paymentTokenCount = []
        for j, tokenVal in enumerate(tokenCountPerT):
            if j == 0: # account for the first transaction which founds the wallet and is not part of the transactions list
                continue
            else:
                j = j - 1
            if transactions[j] < 0:
                paymentTokenCount.append(tokenCountPerT[j+1])
        mean = np.mean(paymentTokenCount)
        std = np.std(paymentTokenCount)
        print("Payment token count length:", len(paymentTokenCount))
        print("Payment token count mean:", mean, "+-", std)
        paymentno = []
        with open('Data/payment_token_count_' + str(i) + '.dat', 'w') as f:
            for k, tokenCount in enumerate(paymentTokenCount):
                f.write(f"{tokenCount}\n")
                paymentno.append(k)

        plt.scatter(paymentno, paymentTokenCount, marker='o', color='black', linewidths=0.05)
        plt.title('Payment Token Count')
        plt.xlabel('Payment Number')
        plt.ylabel('Number of Tokens in Payment')
        plt.savefig('Data/payment_token_count_' + str(i) + '.png')
        plt.clf()  # Clear the current figure

        with open('Data/total_transactions.dat', 'a') as f:
            for t in transactions:
                f.write(f"{t}\n")
                #totalTransaction.append(t)

        with open('Data/total_token_values_.dat', 'a') as f:
            for v in tokenVals:
                f.write(f"{v}\n")
                #alltokenValues.append(v)

        totalValues.append(totalValue)
        totalTokenCounts.append(totalTokensInWallet)
        totalMaxTokenVals.append(maxTokenValRemoved)
        paymentTokenCountMeans.append(mean)
        paymentTokenCountStddev.append(std)
    
    tokenVals = 0 
    maxTokenValRemoved
    totalValue = 0
    totalTokensInWallet = 0
    tokenCountPerT = 0
    tokenCountHistory = 0
    totalValueHistory = 0

    totalTransaction = np.loadtxt('Data/total_transactions.dat')
    alltokenValues = np.loadtxt('Data/total_token_values_.dat')
    plt.hist(totalTransaction, bins=200, density=False)
    plt.title("Histogram of Transactions over all Simulations")
    plt.xlabel("Transaction Value")
    plt.ylabel("Frequency")
    plt.savefig("DataGlobal/histogram_transactions.png")
    plt.clf()  # Clear the current figure

    plt.hist(alltokenValues, bins=200, density=False)
    plt.title("Token Values from all Simulations, Max Token removed")
    plt.xlabel("Token Value")
    plt.ylabel("Frequency")
    plt.savefig("DataGlobal/histogram_token_values.png")
    plt.clf()  # Clear the current figure


    simNoList = list(range(numSimulations))
    with open('DataGlobal/total_values.dat', 'w') as f:
        for v in totalValues:
            f.write(f"{v}\n")
    plt.scatter(simNoList, totalValues, marker='o', color='blue', linewidths=0.05)
    plt.title('Final state: Total Value in Wallet')
    plt.xlabel('Simulation Index')
    #plt.ylabel('Total Value in Wallet')
    plt.savefig('DataGlobal/total_values.png')
    plt.clf()  # Clear the current figure

    with open('DataGlobal/total_token_counts.dat', 'w') as f:
        for c in totalTokenCounts:
            f.write(f"{c}\n")
    plt.scatter(simNoList, totalTokenCounts, marker='o', color='black')
    plt.title('Final State UTXO Pool Size')
    plt.xlabel('Simulation Index')
    #plt.ylabel('Total Token Count in Wallet')
    plt.savefig('DataGlobal/total_token_counts.png')
    plt.clf()  # Clear the current figure


    with open('DataGlobal/total_max_token_vals.dat', 'w') as f:
        for m in totalMaxTokenVals:
            f.write(f"{m}\n")

    plt.scatter(simNoList, totalMaxTokenVals, marker='o', color='black')
    plt.title('Maximal Token Value in Wallet per Simulation')
    plt.xlabel('Simulation Number')
    plt.ylabel('Maximal Token Value in Wallet')
    plt.savefig('DataGlobal/total_max_token_vals.png')
    plt.clf()  # Clear the current figure


    with open('DataGlobal/payment_token_count_means.dat', 'w') as f:
        for m in paymentTokenCountMeans:
            f.write(f"{m}\n")
    with open('DataGlobal/payment_token_count_stds.dat', 'w') as f:
        for s in paymentTokenCountStddev:
            f.write(f"{s}\n")
    plt.errorbar(simNoList, paymentTokenCountMeans, yerr=paymentTokenCountStddev, fmt='o', color='black', capsize=5)
    plt.title('Mean Payment Token Count per Simulation')
    plt.xlabel('Simulation Number')
    plt.ylabel('Mean Payment Token Count')
    plt.savefig('DataGlobal/payment_token_count_means.png')
    plt.clf()  # Clear the current figure


    