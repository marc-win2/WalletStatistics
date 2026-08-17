# Created by Marc Winstel on July 25, 2025
import argparse
import os

import numpy as np 
import matplotlib.pyplot as plt 


def meanAndStd(values):
    """Return the population mean and standard deviation of sample values."""
    return np.mean(values), np.std(values)


def extendWithValues(destination, values):
    """Append a scalar or extend an iterable of values into a result list."""
    convertedValues = np.asarray(values).tolist()
    if isinstance(convertedValues, list):
        destination.extend(convertedValues)
    else:
        destination.append(convertedValues)


def positiveInteger(value):
    """Return a positive integer or raise a command-line validation error."""
    integerValue = int(value)
    if integerValue <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return integerValue


def parseCommandLineArguments(arguments=None):
    """Parse configuration for aggregating simulation output."""
    parser = argparse.ArgumentParser(
        description="Average and plot output from multiple simulation runs."
    )
    parser.add_argument(
        "--num_runs",
        "--num-runs",
        dest="num_runs",
        type=positiveInteger,
        default=100,
        help="number of simulation runs to aggregate (default: 100)",
    )
    parser.add_argument(
        "--num_payments",
        "--num-payments",
        dest="num_payments",
        type=positiveInteger,
        default=100000,
        help="number of payments per run (default: 100000)",
    )
    parser.add_argument(
        "--chunk_size",
        "--chunk-size",
        dest="chunk_size",
        type=positiveInteger,
        default=20000,
        help="rows processed at once from each run (default: 20000)",
    )
    parser.add_argument(
        "--data_path",
        "--data-path",
        dest="data_path",
        default="./Data",
        help="directory containing per-run data (default: ./Data)",
    )
    parser.add_argument(
        "--save_path",
        "--save-path",
        dest="save_path",
        default="./DataGlobal",
        help="directory for aggregate output (default: ./DataGlobal)",
    )
    return parser.parse_args(arguments)




if __name__ == "__main__":
    commandLineArguments = parseCommandLineArguments()
    numSimulations = commandLineArguments.num_runs

    dataPath = commandLineArguments.data_path
    savePath = commandLineArguments.save_path

    transActionNumber = np.atleast_1d(
        np.loadtxt(os.path.join(dataPath, "transaction0.dat"))
    )
    noTransactions = len(transActionNumber)
    print("Number of transactions in the simulation: ", noTransactions)

    avgTotalValues = []
    stdDevTotalValues = []
    
    avgTotalTokensInWallets = []
    stdDevTotalTokensInWallets = []


    allTokenValues = []

    stepSize = commandLineArguments.chunk_size
    # Chunking bounds memory use by loading only one slice from each run at once.
    for transIndex in np.arange(0,noTransactions, stepSize):
        print("Processing transaction index: ", transIndex)
        currentStepSize = min(stepSize, noTransactions - int(transIndex))
        tv_ = [[] for _ in range(currentStepSize)]
        tT_ = [[] for _ in range(currentStepSize)]

        for simIndex in np.arange(numSimulations):
            # Load the data for each simulation
            totalValue = np.atleast_2d(np.genfromtxt(os.path.join(dataPath, "WalletValue_" + str(simIndex) + ".dat"), dtype=float, skip_header=transIndex, max_rows=currentStepSize))
            totalTokensInWallet = np.atleast_2d(np.genfromtxt(os.path.join(dataPath, "TokenCount_" + str(simIndex) + ".dat"), dtype=float, skip_header=transIndex, max_rows=currentStepSize))

            # Store the values to average over all simulations
            for j, tv in enumerate(totalValue):
                tv_[j].append(totalValue[j][1])
                tT_[j].append(totalTokensInWallet[j][1])


        for j in np.arange(currentStepSize):
            avgTotalValue, stdDevTotalValue = meanAndStd(tv_[j])
            avgTotalValues.append(avgTotalValue)
            stdDevTotalValues.append(stdDevTotalValue)

            avgTotalTokensInWallet, stdDevTotalTokensInWallet = meanAndStd(tT_[j])
            avgTotalTokensInWallets.append(avgTotalTokensInWallet)
            stdDevTotalTokensInWallets.append(stdDevTotalTokensInWallet)

    
    totalValue= 0
    totalTokensInWallet = 0
    tv_ = 0
    tT_ = 0

    noPayments = commandLineArguments.num_payments
    print(noPayments, " payments in the simulation")

    avgPaymentTokenCounts = []
    stdDevPaymentTokenCounts = []

    for payIndex in np.arange(0, noPayments, stepSize):
        currentStepSize = min(stepSize, noPayments - int(payIndex))
        avgPaymentTokenCount = [[] for _ in range(currentStepSize)]
        print("Processing payment index: ", payIndex)

        for simIndex in np.arange(numSimulations):
            # Load the payment token count for each simulation
            paymentTokenCount = np.atleast_1d(np.genfromtxt(os.path.join(dataPath, "payment_token_count_" + str(simIndex) + ".dat"), dtype=float, skip_header=payIndex, max_rows=currentStepSize))

            # Store the values to average over all simulations
            for j, ptc in enumerate(paymentTokenCount):
                avgPaymentTokenCount[j].append(ptc)

        for j in np.arange(currentStepSize):
            avgPaymentTokenCountValue, stdDevPaymentTokenCountValue = meanAndStd(
                avgPaymentTokenCount[j]
            )
            avgPaymentTokenCounts.append(avgPaymentTokenCountValue)
            stdDevPaymentTokenCounts.append(stdDevPaymentTokenCountValue)

    print(len(avgPaymentTokenCounts), " average payment token counts")

    for simIndex in np.arange(numSimulations):

        # Load the token values for each simulation
        tokenValues = np.genfromtxt(os.path.join(dataPath, "token_values_" + str(simIndex) + ".dat"), dtype=float)

        extendWithValues(allTokenValues, tokenValues)
    maxTokens = np.genfromtxt(os.path.join(savePath, "total_max_token_vals.dat"), dtype=float)
    extendWithValues(allTokenValues, maxTokens)

    np.savetxt(os.path.join(savePath, "all_token_values.dat"), allTokenValues)
    # histrogram of all token values crosscheck
    plt.hist(allTokenValues, bins=200, density=False, alpha=0.7, label='All Token Values')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Token Values')
    plt.savefig(os.path.join(savePath, "histogram_all_token_values_crosscheck.png"))
    plt.clf()  # Clear the current figure for the next plot

    zoomedTokenValues = [val for val in allTokenValues if val < 5000]
    plt.hist(zoomedTokenValues, bins=200, density=False, alpha=0.7, label='All Token Values')
    plt.xlabel('Token Value')
    plt.savefig(os.path.join(savePath, "histogram_token_values_crosscheck_zoomed.png"))
    plt.clf()  # Clear the current figure for the next plot

    np.savetxt(os.path.join(savePath, "avg_total_values.dat"), avgTotalValues)
    np.savetxt(os.path.join(savePath, "std_dev_total_values.dat"), stdDevTotalValues)

    np.savetxt(os.path.join(savePath, "avg_total_tokens_in_wallets.dat"), avgTotalTokensInWallets)
    np.savetxt(os.path.join(savePath, "std_dev_total_tokens_in_wallets.dat"), stdDevTotalTokensInWallets)

    np.savetxt(os.path.join(savePath, "avg_payment_token_counts.dat"), avgPaymentTokenCounts)
    np.savetxt(os.path.join(savePath, "std_dev_payment_token_counts.dat"), stdDevPaymentTokenCounts)


    # Plot the average total values
    plt.scatter(np.arange(noTransactions), avgTotalValues, label='Average Total Value') # yerr=stdDevTotalValues
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Value')
    plt.title('Average Total Values Over Transactions')
    plt.savefig(os.path.join(savePath, "avg_total_values_over_transactions.png"))
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average total tokens in wallets
    plt.scatter(np.arange(noTransactions), avgTotalTokensInWallets, label='Average Total Tokens in Wallets')# yerr=stdDevTotalTokensInWallets
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Total Tokens in Wallets')
    plt.title('Average Total Tokens in Wallets Over Transactions')
    plt.savefig(os.path.join(savePath, "avg_total_tokens_in_wallets_over_transactions.png"))
    plt.clf()  # Clear the current figure for the next plot

    # Plot the average payment token counts
    plt.scatter(np.arange(noPayments), avgPaymentTokenCounts, label='Average Payment Token Count') # 
    plt.xlabel('Transaction Index')
    plt.ylabel('Average Payment Token Count')
    plt.title('Average Payment Token Counts Over Transactions')
    plt.savefig(os.path.join(savePath, "avg_payment_token_counts_over_transactions.png"))
    plt.clf()  # Clear the current figure for the next plot
