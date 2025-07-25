# Created by Marc Winstel on July 25, 2025
import numpy as np 
import matplotlib as plt 




if __name__ == "__main__":
    numSimulations = 50

    dataPath = "./Data/"
    savePath = "./DataGlobal/"

    transActionNumber = np.loadtxt(dataPath + "transaction0.dat")
    noTransactions = len(transActionNumber)
    print("Number of transactions in the simulation: ", noTransactions)

   

    avgTotalValues = []
    stdDevTotalValues = []
    
    avgTotalTokensInWallets = []
    stdDevTotalTokensInWallets = []

    avgPaymentTokenCounts = []
    stdDevPaymentTokenCounts = []

    allTokenValues = []
    # Initialize arrays to hold the data for each simulation    
    for transIndex in np.arange(noTransactions):
        if transIndex % 200 == 0:
            print("Processing transaction index: ", transIndex)
        avgTotalValue = []

        avgTotalTokensInWallet = []

        avgPaymentTokenCount = []

        for simIndex in np.arange(numSimulations):
            # Load the data for each simulation
            totalValue = np.genfromtxt(dataPath + "WalletValue_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=1)
            totalTokensInWallet = np.genfromtxt(dataPath + "TokenCount_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=1)
            paymentTokenCount = np.genfromtxt(dataPath + "payment_token_count_" + str(simIndex) + ".dat", dtype=float, skip_header=transIndex, max_rows=1)

            
            # Store the values to average over all simulations
            avgTotalValue.append(totalValue)
            avgTotalTokensInWallet.append(totalTokensInWallet)
            avgPaymentTokenCount.append(paymentTokenCount)
        

        avgTotalValues.append(np.mean(avgTotalValue))
        stdDevTotalValues.append(np.std(avgTotalValue))

        avgTotalTokensInWallets.append(np.mean(avgTotalTokensInWallet))
        stdDevTotalTokensInWallets.append(np.std(avgTotalTokensInWallet))

        avgPaymentTokenCounts.append(np.mean(avgPaymentTokenCount))
        stdDevPaymentTokenCounts.append(np.std(avgPaymentTokenCount))


    for simIndex in np.arange(numSimulations):

        # Load the token values for each simulation
        tokenValues = np.genfromtxt(dataPath + "TokenValues_" + str(simIndex) + ".dat", dtype=float)
        
        # Append the token values to the allTokenValues list
        allTokenValues.extend(tokenValues)

    np.savetxt(savePath + "all_token_values.dat", allTokenValues)
    # histrogram of all token values crosscheck
    plt.hist(allTokenValues, bins=200, density=False, alpha=0.7, label='All Token Values')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Token Values')
    plt.savefig(savePath + "histogram_all_token_values_crosscheck.png")
    plt.clf()  # Clear the current figure for the next plot



    np.savetxt(savePath + "avg_total_values.dat", avgTotalValues)
    np.savetxt(savePath + "std_dev_total_values.dat", stdDevTotalValues)

    np.savetxt(savePath + "avg_total_tokens_in_wallets.dat", avgTotalTokensInWallets)
    np.savetxt(savePath + "std_dev_total_tokens_in_wallets.dat", stdDevTotalTokensInWallets)

    np.savetxt(savePath + "avg_payment_token_counts.dat", avgPaymentTokenCounts)
    np.savetxt(savePath + "std_dev_payment_token_counts.dat", stdDevPaymentTokenCounts)


    # Plot the average total values
    plt.errorbar(np.arange(noTransactions), avgTotalValues, yerr=stdDevTotalValues, fmt='o', label='Average Total Value')
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Value')
    plt.title('Average Total Values Over Transactions')
    plt.savefig(savePath + "avg_total_values_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average total tokens in wallets
    plt.errorbar(np.arange(noTransactions), avgTotalTokensInWallets, yerr=stdDevTotalTokensInWallets, fmt='o', label='Average Total Tokens in Wallets')
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Tokens in Wallets')
    plt.title('Average Total Tokens in Wallets Over Transactions')
    plt.savefig(savePath + "avg_total_tokens_in_wallets_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average payment token counts
    plt.errorbar(np.arange(noTransactions), avgPaymentTokenCounts, yerr=stdDevPaymentTokenCounts, fmt='o', label='Average Payment Token Count')
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Payment Token Count')
    plt.title('Average Payment Token Counts Over Transactions')
    plt.savefig(savePath + "avg_payment_token_counts_over_transactions.png")
    plt.clf()  # Clear the current figure for the next plot