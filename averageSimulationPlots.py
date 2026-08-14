# Created by Marc Winstel on July 25, 2025
import numpy as np 
import matplotlib.pyplot as plt 




if __name__ == "__main__":
    numSimulations = 100

    addPrefix = "."# "./Simulations/Gaussian_Boltzmannadjustbeta_normalrefund" #"Dirichlet_BoltzmannBetaadjusted_noinitialfunding1"#"Dirichtlet_canonicaladjustbeta_run1"
    dataPath = addPrefix + "/Data/"
    savePath = addPrefix + "/DataGlobal/"

    transActionNumber = np.loadtxt(dataPath + "transaction0.dat")
    noTransactions = len(transActionNumber)
    print("Number of transactions in the simulation: ", noTransactions)

    avgTotalValues = []
    stdDevTotalValues = []
    
    avgTotalTokensInWallets = []
    stdDevTotalTokensInWallets = []


    allTokenValues = []

    stepSize = 20000
    # Initialize arrays to hold the data for each simulation    
    for transIndex in np.arange(0,noTransactions, stepSize):
        print("Processing transaction index: ", transIndex)
        avgTotalValue = []*stepSize
        
        avgTotalTokensInWallet = []*stepSize

        #avgPaymentTokenCount = []*stepSize

        tv_ = [[] for _ in range(stepSize)]
        tT_ = [[] for _ in range(stepSize)]
        #tP_ = [[] for _ in range(stepSize // 4)]

        for simIndex in np.arange(numSimulations):
            # Load the data for each simulation
            totalValue = np.genfromtxt(dataPath + "WalletValue_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=stepSize)
            totalTokensInWallet = np.genfromtxt(dataPath + "TokenCount_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=stepSize)
            #paymentTokenCount = np.genfromtxt(dataPath + "payment_token_count_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=stepSize)
            #print("Total Value: ", totalValue   )

            # Store the values to average over all simulations
            for j, tv in enumerate(totalValue):
                tv_[j].append(totalValue[j][1])
                tT_[j].append(totalTokensInWallet[j][1])
                #tP_[j].append(paymentTokenCount[j])


        for j in np.arange(stepSize):
            avgTotalValues.append(np.mean(tv_[j]))
            stdDevTotalValues.append(np.std(tv_[j]))

            avgTotalTokensInWallets.append(np.mean(tT_[j]))
            stdDevTotalTokensInWallets.append(np.std(tT_[j]))

            #avgPaymentTokenCounts.append(np.mean(tP_[j]))
            #stdDevPaymentTokenCounts.append(np.std(tP_[j]))
    
    totalValue= 0
    totalTokensInWallet = 0
    tv_ = 0
    tT_ = 0

    noPayments = 100000
    print(noPayments, " payments in the simulation")

    avgPaymentTokenCounts = []
    stdDevPaymentTokenCounts = []

    for payIndex in np.arange(0, noPayments, stepSize):
        avgPaymentTokenCount = [[] for _ in range(stepSize)]
        stdDevPaymentTokenCount = [[] for _ in range(stepSize)]
        print("Processing payment index: ", payIndex)

        for simIndex in np.arange(numSimulations):
            # Load the payment token count for each simulation
            paymentTokenCount = np.genfromtxt(dataPath + "payment_token_count_" + str(simIndex) + ".dat", dtype=float, skip_header=payIndex, max_rows=stepSize)

            # Store the values to average over all simulations
            for j, ptc in enumerate(paymentTokenCount):
                avgPaymentTokenCount[j].append(ptc)

        for j in np.arange(stepSize):
            avgPaymentTokenCounts.append(np.mean(avgPaymentTokenCount[j]))
            stdDevPaymentTokenCounts.append(np.std(avgPaymentTokenCount[j]))

    print("Average Payment Token Counts: ", avgPaymentTokenCounts)
    print(len(avgPaymentTokenCounts), " average payment token counts")

    for simIndex in np.arange(numSimulations):

        # Load the token values for each simulation
        tokenValues = np.genfromtxt(dataPath + "token_values_" + str(simIndex) + ".dat", dtype=float)
        
        tokenValues = tokenValues.tolist()
        #print(tokenValues)
        
        # Check if the tokenValues is
        # Append the token values to the allTokenValues list
        # check if tokenValues is a list or a single value
        if isinstance(tokenValues, list):
            for t in tokenValues:
                allTokenValues.append(t)
        else:
            allTokenValues.append(tokenValues)
    maxTokens = np.genfromtxt(savePath + "total_max_token_vals.dat", dtype=float)
    allTokenValues.extend(maxTokens.tolist())

    np.savetxt(savePath + "all_token_values.dat", allTokenValues)
    # histrogram of all token values crosscheck
    plt.hist(allTokenValues, bins=200, density=False, alpha=0.7, label='All Token Values')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Token Values')
    plt.savefig(savePath + "histogram_all_token_values_crosscheck.png")
    plt.clf()  # Clear the current figure for the next plot

    zoomedTokenValues = [val for val in allTokenValues if val < 5000]
    plt.hist(zoomedTokenValues, bins=200, density=False, alpha=0.7, label='All Token Values')
    plt.xlabel('Token Value')
    plt.savefig(savePath + "histogram_token_values_crosscheck_zoomed.png")
    plt.clf()  # Clear the current figure for the next plot

    np.savetxt(savePath + "avg_total_values.dat", avgTotalValues)
    np.savetxt(savePath + "std_dev_total_values.dat", stdDevTotalValues)

    np.savetxt(savePath + "avg_total_tokens_in_wallets.dat", avgTotalTokensInWallets)
    np.savetxt(savePath + "std_dev_total_tokens_in_wallets.dat", stdDevTotalTokensInWallets)

    np.savetxt(savePath + "avg_payment_token_counts.dat", avgPaymentTokenCounts)
    np.savetxt(savePath + "std_dev_payment_token_counts.dat", stdDevPaymentTokenCounts)


    # Plot the average total values
    plt.scatter(np.arange(noTransactions), avgTotalValues, label='Average Total Value') # yerr=stdDevTotalValues
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Value')
    plt.title('Average Total Values Over Transactions')
    plt.savefig(savePath + "avg_total_values_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average total tokens in wallets
    plt.scatter(np.arange(noTransactions), avgTotalTokensInWallets, label='Average Total Tokens in Wallets')# yerr=stdDevTotalTokensInWallets
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Tokens in Wallets')
    plt.title('Average Total Tokens in Wallets Over Transactions')
    plt.savefig(savePath + "avg_total_tokens_in_wallets_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average payment token counts
    plt.scatter(np.arange(noPayments), avgPaymentTokenCounts, label='Average Payment Token Count') # 
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Payment Token Count')
    plt.title('Average Payment Token Counts Over Transactions')
    plt.savefig(savePath + "avg_payment_token_counts_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot