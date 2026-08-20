# Created by Marc Winstel on 14.07.25
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import os
from transaction import RandomTransactionGenerator
from coinselection import CoinSelectionDistribution
from wallet import Token, Wallet
from simulation import SimulationHandler


TRANSACTION_SCENARIO_SEED_IDS = {
    'gaussian': 0,
    'dirichletFloat': 1,
}
BETA_ADJUSTMENT_SEED_IDS = {
    'legacy': 0,
    'microcanonicalExact': 1,
    'microcanonicalApprox': 2,
}


def deriveSeed(rootSeed, *streamCoordinates):
    """Derive a stable independent integer seed from a root seed."""
    if rootSeed is None:
        return None
    seedSequence = np.random.SeedSequence([rootSeed, *streamCoordinates])
    return int(seedSequence.generate_state(1, dtype=np.uint64)[0])


def deriveRunSeeds(rootSeed, transactionScenario, betaAdjustmentMode, runIndex):
    """Derive workload and simulation seeds for one experiment run."""
    if rootSeed is None:
        return None, None
    if transactionScenario not in TRANSACTION_SCENARIO_SEED_IDS:
        raise ValueError(
            "Invalid transaction scenario. Choose from gaussian or "
            "dirichletFloat."
        )
    if betaAdjustmentMode not in BETA_ADJUSTMENT_SEED_IDS:
        raise ValueError(
            "Invalid beta adjustment mode. Choose from legacy, "
            "microcanonicalExact, or microcanonicalApprox."
        )
    scenarioId = TRANSACTION_SCENARIO_SEED_IDS[transactionScenario]
    betaModeId = BETA_ADJUSTMENT_SEED_IDS[betaAdjustmentMode]

    # Excluding betaModeId keeps each scenario's input transactions identical
    # across beta modes, while coin-selection randomness remains independent.
    transactionSeed = deriveSeed(rootSeed, 0, scenarioId, runIndex)
    simulationSeed = deriveSeed(rootSeed, 1, scenarioId, betaModeId, runIndex)
    return transactionSeed, simulationSeed


def plottingTransactionsTest():
    transactionGenerator = RandomTransactionGenerator()
    
    transactionGenerator.maxAbsTransactionValue = 10**6### maximal, absolute transaction value
    
    
    #transactionGenerator.plotGaussianTransactionTestOne(10**6)
    
    transactions = transactionGenerator.plotGaussianWithUniformOutliersTransactionTestOne(10**5)
    print(transactions.mean(), transactions.std())


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




def plotTransactionData(
    transactions,
    payments,
    deposits,
    plottingIndex=0,
    dataDirectory='Data',
):
    plt.hist(deposits, bins=200, density=False, alpha=0.7, color='green', label='Deposits')
    plt.hist(payments, bins=200, density=False, alpha=0.7, color='red', label='Payments')
    plt.title('Transactions Histogram')
    plt.xlabel('Value')
    plt.legend()
    plt.savefig(os.path.join(dataDirectory, 'transactions_histogram' + str(plottingIndex) + '.png'))
    plt.clf()  # Clear the current figure
    with open(os.path.join(dataDirectory, 'transaction' + str(plottingIndex) + '.dat'), 'w') as f:
        for t in transactions:
            f.write(f"{t}\n")


def writeValues(filePath, values, mode='w'):
    """Write one value per line using the requested file mode."""
    with open(filePath, mode) as f:
        for value in values:
            f.write(f"{value}\n")


def writeIndexedValues(filePath, indices, values):
    """Write an index and its corresponding value on each line."""
    with open(filePath, 'w') as f:
        for index, value in zip(indices, values):
            f.write(f"{index} {value}\n")


def saveIndexedHistory(
    indices,
    values,
    fileStem,
    simulationIndex,
    marker,
    color,
    yLabel,
    title=None,
    yScale=None,
    dataDirectory='Data',
):
    """Persist and plot a transaction-indexed simulation history."""
    outputBase = os.path.join(dataDirectory, fileStem + '_' + str(simulationIndex))
    writeIndexedValues(outputBase + '.dat', indices, values)
    plt.scatter(indices, values, marker=marker, color=color, linewidths=0.05)
    if yScale is not None:
        plt.yscale(yScale)
    if title is not None:
        plt.title(title)
    plt.xlabel('Transaction Index')
    plt.ylabel(yLabel)
    plt.savefig(outputBase + '.png')
    plt.clf()  # Clear the current figure


def saveFinalTokenValues(simulation, simulationIndex, dataDirectory='Data'):
    """Save final token values while keeping the maximal token separate."""
    maxTokenValue = max(token.value for token in simulation.highThroughputWallet.tokens)
    print("Maximal token value in wallet:", maxTokenValue)

    tokenValues = [token.value for token in simulation.highThroughputWallet.tokens]
    tokenValues.remove(maxTokenValue)

    plt.hist(tokenValues, bins=200, density=False)
    plt.title("Token values in Wallet after a single run")
    plt.xlabel("Token Value")
    plt.savefig(os.path.join(dataDirectory, 'token_values_histogram_' + str(simulationIndex) + '.png'))
    writeValues(
        os.path.join(dataDirectory, 'token_values_' + str(simulationIndex) + '.dat'),
        tokenValues,
    )
    return tokenValues, maxTokenValue


def saveSimulationHistories(
    simulation,
    transactions,
    simulationIndex,
    dataDirectory='Data',
):
    """Save wallet value, token count, beta, and refund histories."""
    transactionIndices = np.arange(len(transactions) + 1)
    saveIndexedHistory(
        transactionIndices,
        simulation.totalValueHistory,
        'WalletValue',
        simulationIndex,
        marker='o',
        color='blue',
        yLabel='Total Value in Wallet',
        dataDirectory=dataDirectory,
    )
    saveIndexedHistory(
        transactionIndices,
        simulation.tokenCountHistory,
        'TokenCount',
        simulationIndex,
        marker='x',
        color='blue',
        yLabel='UTXO Pool Size',
        dataDirectory=dataDirectory,
    )

    betaScale = 'linear' if np.any(simulation.saveBetaHistory) < 1e-10 else 'log'
    saveIndexedHistory(
        transactionIndices,
        simulation.saveBetaHistory,
        'BetaPerTransaction',
        simulationIndex,
        marker='o',
        color='black',
        yLabel='Beta Value',
        title='Beta Value per Transaction',
        yScale=betaScale,
        dataDirectory=dataDirectory,
    )
    writeValues(
        os.path.join(
            dataDirectory,
            'emergence_refund_indices_' + str(simulationIndex) + '.dat',
        ),
        simulation.emergenceRefundTransactionIndices,
    )


def singleSimulation(
    transactions,
    tokenDenominationBuckets,
    simulationIndex,
    drawDeposit=False,
    adjustBeta=True,
    doEmergRefund=True,
    useBuckets=False,
    mode="canonical",
    betaAdjustmentMode="legacy",
    dataDirectory='Data',
    seed=None,
    coinSelectionStrategy="boltzmann",
    samplingMode="token",
):
    simulation = SimulationHandler(
        tokenDenominationBuckets=tokenDenominationBuckets,
        beta=1e-03,
        drawDepositToken=drawDeposit,
        adjustBetaAfterEachTransaction=adjustBeta,
        doEmergRefund=doEmergRefund,
        useBucketsForProbabilityComp=useBuckets,
        mode=mode,
        betaAdjustmentMode=betaAdjustmentMode,
        seed=seed,
        coinSelectionStrategy=coinSelectionStrategy,
        samplingMode=samplingMode,
    )
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
    tokenValues, maxTokenValue = saveFinalTokenValues(
        simulation,
        simulationIndex,
        dataDirectory=dataDirectory,
    )
    saveSimulationHistories(
        simulation,
        transactions,
        simulationIndex,
        dataDirectory=dataDirectory,
    )

    return tokenValues, maxTokenValue, simulation.highThroughputWallet.getTotalValue(), simulation.highThroughputWallet.getTokenCount(), simulation.tokenCountInvolvedInTransaction, simulation.tokenCountHistory, simulation.totalValueHistory




def generateDoubleGaussianTransactionsAndPlotThem(
    plottingIndex=0,
    noPayments=100000,
    xFactor=3,
    dataDirectory='Data',
    seed=None,
):
    transactionGenerator = RandomTransactionGenerator(seed=seed)
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

    plotTransactionData(
        transactions,
        payments,
        deposits,
        plottingIndex,
        dataDirectory=dataDirectory,
    )

    
    return transactions, deposits, payments

def generateTransactions_PaymentsDirichlet_AndPlotThem(
    plottingIndex=0,
    noDeposits=100000,
    xFactor=10,
    generateDirichletAsFloats=True,
    dataDirectory='Data',
    seed=None,
):

    print("Generating Dirichlet Payments and constant deposits")
    deposits = [2000] * noDeposits
    payments = []
    transactionGenerator = RandomTransactionGenerator(seed=seed)
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

    plotTransactionData(
        transactions,
        payments,
        deposits,
        plottingIndex,
        dataDirectory=dataDirectory,
    )

    return transactions, deposits, payments


def prepareOutputDirectory(directoryPath, existingMessages):
    """Create an output directory or request confirmation before reuse."""
    if os.path.isdir(directoryPath) is False:
        os.mkdir(directoryPath)
        return

    for message in existingMessages:
        print(message)
    try:
        input("Press anything to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        exit(0)


def prepareOutputDirectories(
    dataDirectory='Data',
    globalDataDirectory='DataGlobal',
):
    """Prepare the per-run and aggregate output directories."""
    prepareOutputDirectory(
        dataDirectory,
        ["Data directory already exists, overwriting data files."],
    )
    prepareOutputDirectory(
        globalDataDirectory,
        [
            "DataGlobal directory already exists, appending to the data files.",
            "Data Global files are always appended to, so the overall data might contain results from previous runs. Please check that this directory is empty before running the script.",
        ],
    )


def getPaymentTokenCounts(
    transactions,
    tokenCountPerTransaction,
    includeZeroValuePayments=False,
):
    """Extract payment token counts while excluding initial funding."""
    paymentTokenCounts = []
    for transactionIndex, tokenCount in enumerate(tokenCountPerTransaction):
        if transactionIndex == 0:
            continue
        transactionIndex -= 1
        transactionValue = transactions[transactionIndex]
        if transactionValue < 0 or (
            includeZeroValuePayments and transactionValue == 0
        ):
            paymentTokenCounts.append(tokenCountPerTransaction[transactionIndex + 1])
    return paymentTokenCounts


def savePaymentTokenCounts(
    paymentTokenCounts,
    simulationIndex,
    dataDirectory='Data',
):
    """Persist and plot payment token counts and return their summary statistics."""
    mean = np.mean(paymentTokenCounts)
    std = np.std(paymentTokenCounts)
    print("Payment token count length:", len(paymentTokenCounts))
    print("Payment token count mean:", mean, "+-", std)

    paymentNumbers = []
    outputBase = os.path.join(
        dataDirectory,
        'payment_token_count_' + str(simulationIndex),
    )
    with open(outputBase + '.dat', 'w') as f:
        for paymentNumber, tokenCount in enumerate(paymentTokenCounts):
            f.write(f"{tokenCount}\n")
            paymentNumbers.append(paymentNumber)

    plt.scatter(paymentNumbers, paymentTokenCounts, marker='o', color='black', linewidths=0.05)
    plt.title('Payment Token Count')
    plt.xlabel('Payment Number')
    plt.ylabel('Number of Tokens in Payment')
    plt.savefig(outputBase + '.png')
    plt.clf()  # Clear the current figure
    return mean, std


def appendSimulationData(
    transactions,
    tokenValues,
    dataDirectory='Data',
):
    """Append one run's transactions and final token values to aggregate files."""
    writeValues(
        os.path.join(dataDirectory, 'total_transactions.dat'),
        transactions,
        mode='a',
    )
    writeValues(
        os.path.join(dataDirectory, 'total_token_values_.dat'),
        tokenValues,
        mode='a',
    )


def generateTransactionScenario(
    transactionScenario,
    simulationIndex,
    noPayments,
    dataDirectory='Data',
    seed=None,
):
    """Generate one of the two workloads used by the experiment matrix."""
    if transactionScenario == 'gaussian':
        return generateDoubleGaussianTransactionsAndPlotThem(
            plottingIndex=simulationIndex,
            noPayments=noPayments,
            xFactor=3,
            dataDirectory=dataDirectory,
            seed=seed,
        )

    if transactionScenario == 'dirichletFloat':
        paymentsPerDeposit = 10
        if noPayments % paymentsPerDeposit != 0:
            raise ValueError(
                "Dirichlet payment count must be divisible by payments per deposit."
            )
        return generateTransactions_PaymentsDirichlet_AndPlotThem(
            plottingIndex=simulationIndex,
            noDeposits=noPayments // paymentsPerDeposit,
            xFactor=paymentsPerDeposit,
            generateDirichletAsFloats=True,
            dataDirectory=dataDirectory,
            seed=seed,
        )

    raise ValueError(
        "Invalid transaction scenario. Choose from gaussian or dirichletFloat."
    )


def runSimulationBatch(
    numSimulations,
    tokenDenominationBuckets,
    transactionScenario='gaussian',
    betaAdjustmentMode='legacy',
    noPayments=100000,
    dataDirectory='Data',
    seed=None,
    coinSelectionStrategy='boltzmann',
    samplingMode='token',
):
    """Run the configured simulation batch and collect final-state summaries."""
    totalValues = []
    totalTokenCounts = []
    totalMaxTokenValues = []
    paymentTokenCountMeans = []
    paymentTokenCountStddevs = []

    for simulationIndex in range(numSimulations):
        transactionSeed, simulationSeed = deriveRunSeeds(
            seed,
            transactionScenario,
            betaAdjustmentMode,
            simulationIndex,
        )
        transactions, deposits, payments = generateTransactionScenario(
            transactionScenario,
            simulationIndex,
            noPayments,
            dataDirectory=dataDirectory,
            seed=transactionSeed,
        )
        tokenValues, maxTokenValue, totalValue, totalTokensInWallet, tokenCountPerTransaction, tokenCountHistory, totalValueHistory = singleSimulation(
            transactions,
            tokenDenominationBuckets,
            simulationIndex,
            drawDeposit=False,
            adjustBeta=True,
            doEmergRefund=False,
            useBuckets=False,
            mode="canonical",
            betaAdjustmentMode=betaAdjustmentMode,
            dataDirectory=dataDirectory,
            seed=simulationSeed,
            coinSelectionStrategy=coinSelectionStrategy,
            samplingMode=samplingMode,
        )

        paymentTokenCounts = getPaymentTokenCounts(
            transactions,
            tokenCountPerTransaction,
            includeZeroValuePayments=(transactionScenario == 'dirichletFloat'),
        )
        paymentMean, paymentStddev = savePaymentTokenCounts(
            paymentTokenCounts,
            simulationIndex,
            dataDirectory=dataDirectory,
        )
        appendSimulationData(
            transactions,
            tokenValues,
            dataDirectory=dataDirectory,
        )

        totalValues.append(totalValue)
        totalTokenCounts.append(totalTokensInWallet)
        totalMaxTokenValues.append(maxTokenValue)
        paymentTokenCountMeans.append(paymentMean)
        paymentTokenCountStddevs.append(paymentStddev)

    return (
        totalValues,
        totalTokenCounts,
        totalMaxTokenValues,
        paymentTokenCountMeans,
        paymentTokenCountStddevs,
    )


def saveAggregateHistograms(
    dataDirectory='Data',
    globalDataDirectory='DataGlobal',
):
    """Plot aggregate transactions and token values accumulated across runs."""
    totalTransactions = np.loadtxt(
        os.path.join(dataDirectory, 'total_transactions.dat')
    )
    allTokenValues = np.loadtxt(
        os.path.join(dataDirectory, 'total_token_values_.dat')
    )

    plt.hist(totalTransactions, bins=200, density=False)
    plt.title("Histogram of Transactions over all Simulations")
    plt.xlabel("Transaction Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(globalDataDirectory, "histogram_transactions.png"))
    plt.clf()  # Clear the current figure

    plt.hist(allTokenValues, bins=200, density=False)
    plt.title("Token Values from all Simulations, Max Token removed")
    plt.xlabel("Token Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(globalDataDirectory, "histogram_token_values.png"))
    plt.clf()  # Clear the current figure


def saveScatterSummary(
    simulationNumbers,
    values,
    fileStem,
    title,
    xLabel,
    color,
    yLabel=None,
    linewidths=None,
    globalDataDirectory='DataGlobal',
):
    """Persist and plot one final-state value for every simulation."""
    outputBase = os.path.join(globalDataDirectory, fileStem)
    writeValues(outputBase + '.dat', values)
    if linewidths is None:
        plt.scatter(simulationNumbers, values, marker='o', color=color)
    else:
        plt.scatter(
            simulationNumbers,
            values,
            marker='o',
            color=color,
            linewidths=linewidths,
        )
    plt.title(title)
    plt.xlabel(xLabel)
    if yLabel is not None:
        plt.ylabel(yLabel)
    plt.savefig(outputBase + '.png')
    plt.clf()  # Clear the current figure


def saveGlobalSimulationSummaries(
    numSimulations,
    totalValues,
    totalTokenCounts,
    totalMaxTokenValues,
    paymentTokenCountMeans,
    paymentTokenCountStddevs,
    globalDataDirectory='DataGlobal',
):
    """Save final-state and payment-count summaries for the complete batch."""
    simulationNumbers = list(range(numSimulations))
    saveScatterSummary(
        simulationNumbers,
        totalValues,
        'total_values',
        'Final state: Total Value in Wallet',
        'Simulation Index',
        color='blue',
        linewidths=0.05,
        globalDataDirectory=globalDataDirectory,
    )
    saveScatterSummary(
        simulationNumbers,
        totalTokenCounts,
        'total_token_counts',
        'Final State UTXO Pool Size',
        'Simulation Index',
        color='black',
        globalDataDirectory=globalDataDirectory,
    )
    saveScatterSummary(
        simulationNumbers,
        totalMaxTokenValues,
        'total_max_token_vals',
        'Maximal Token Value in Wallet per Simulation',
        'Simulation Number',
        color='black',
        yLabel='Maximal Token Value in Wallet',
        globalDataDirectory=globalDataDirectory,
    )

    writeValues(
        os.path.join(globalDataDirectory, 'payment_token_count_means.dat'),
        paymentTokenCountMeans,
    )
    writeValues(
        os.path.join(globalDataDirectory, 'payment_token_count_stds.dat'),
        paymentTokenCountStddevs,
    )
    plt.errorbar(
        simulationNumbers,
        paymentTokenCountMeans,
        yerr=paymentTokenCountStddevs,
        fmt='o',
        color='black',
        capsize=5,
    )
    plt.title('Mean Payment Token Count per Simulation')
    plt.xlabel('Simulation Number')
    plt.ylabel('Mean Payment Token Count')
    plt.savefig(os.path.join(globalDataDirectory, 'payment_token_count_means.png'))
    plt.clf()  # Clear the current figure


def getBetaAdjustmentExperimentConfigurations():
    """Return all 3 beta modes crossed with the 2 selected workloads."""
    transactionScenarios = (
        ('Gaussian', 'gaussian'),
        ('DirichletFloat', 'dirichletFloat'),
    )
    betaAdjustmentModes = (
        'legacy',
        'microcanonicalExact',
        'microcanonicalApprox',
    )

    configurations = []
    for scenarioDirectoryName, transactionScenario in transactionScenarios:
        for betaAdjustmentMode in betaAdjustmentModes:
            configurations.append(
                {
                    'directoryName': (
                        scenarioDirectoryName + '_' + betaAdjustmentMode
                    ),
                    'transactionScenario': transactionScenario,
                    'betaAdjustmentMode': betaAdjustmentMode,
                }
            )
    return configurations


def runStandaloneSimulationExperiment(
    tokenDenominationBuckets,
    dataDirectory='Data',
    globalDataDirectory='DataGlobal',
    numSimulations=100,
    noPayments=100000,
    transactionScenario='gaussian',
    betaAdjustmentMode='legacy',
    seed=None,
    coinSelectionStrategy='boltzmann',
    samplingMode='token',
):
    """Run one complete scenario using the former standalone output layout."""
    prepareOutputDirectories(dataDirectory, globalDataDirectory)
    simulationSummaries = runSimulationBatch(
        numSimulations,
        tokenDenominationBuckets,
        transactionScenario=transactionScenario,
        betaAdjustmentMode=betaAdjustmentMode,
        noPayments=noPayments,
        dataDirectory=dataDirectory,
        seed=seed,
        coinSelectionStrategy=coinSelectionStrategy,
        samplingMode=samplingMode,
    )
    saveAggregateHistograms(dataDirectory, globalDataDirectory)
    saveGlobalSimulationSummaries(
        numSimulations,
        *simulationSummaries,
        globalDataDirectory=globalDataDirectory,
    )
    return simulationSummaries


def runBetaAdjustmentExperiment(
    outputRoot,
    configuration,
    tokenDenominationBuckets,
    numSimulations=100,
    noPayments=100000,
    seed=None,
    coinSelectionStrategy='boltzmann',
    samplingMode='token',
):
    """Run one workload/beta combination in its own output directory."""
    experimentDirectory = os.path.join(
        outputRoot,
        configuration['directoryName'],
    )
    os.makedirs(experimentDirectory, exist_ok=True)
    dataDirectory = os.path.join(experimentDirectory, 'Data')
    globalDataDirectory = os.path.join(experimentDirectory, 'DataGlobal')

    print(
        "Running experiment:",
        configuration['transactionScenario'],
        configuration['betaAdjustmentMode'],
    )
    runStandaloneSimulationExperiment(
        tokenDenominationBuckets,
        dataDirectory=dataDirectory,
        globalDataDirectory=globalDataDirectory,
        numSimulations=numSimulations,
        noPayments=noPayments,
        transactionScenario=configuration['transactionScenario'],
        betaAdjustmentMode=configuration['betaAdjustmentMode'],
        seed=seed,
        coinSelectionStrategy=coinSelectionStrategy,
        samplingMode=samplingMode,
    )
    return experimentDirectory


def runBetaAdjustmentExperimentMatrix(
    tokenDenominationBuckets,
    outputRoot='Simulations/BetaAdjustmentMatrix',
    numSimulations=100,
    noPayments=100000,
    seed=None,
    coinSelectionStrategy='boltzmann',
    samplingMode='token',
):
    """Run all six workload/beta combinations in isolated directories."""
    os.makedirs(outputRoot, exist_ok=True)
    experimentDirectories = []
    for configuration in getBetaAdjustmentExperimentConfigurations():
        experimentDirectories.append(
            runBetaAdjustmentExperiment(
                outputRoot,
                configuration,
                tokenDenominationBuckets,
                numSimulations=numSimulations,
                noPayments=noPayments,
                seed=seed,
                coinSelectionStrategy=coinSelectionStrategy,
                samplingMode=samplingMode,
            )
        )
    return experimentDirectories


def paymentIterationCount(value):
    """Validate a payment count for both experiment-matrix workloads."""
    numberOfPayments = int(value)
    if numberOfPayments <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    if numberOfPayments % 10 != 0:
        raise argparse.ArgumentTypeError(
            "must be divisible by 10 for the Dirichlet workload"
        )
    return numberOfPayments


def randomSeed(value):
    """Validate a non-negative random seed supplied on the command line."""
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return seed


def parseCommandLineArguments(arguments=None):
    """Parse command-line options for the experiment matrix."""
    parser = argparse.ArgumentParser(
        description="Run the Boltzmann Draw beta-adjustment experiment matrix."
    )
    parser.add_argument(
        "--num_iter",
        type=paymentIterationCount,
        default=100,
        help="number of payments per simulation run (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=randomSeed,
        default=None,
        help="optional root seed for reproducible simulations (default: random)",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    """Execute the six-run beta-adjustment experiment matrix."""
    commandLineArguments = parseCommandLineArguments(arguments)
    tokens = [10**i for i in range(-2, 10)]
    global tokenDenominationBuckets
    tokenDenominationBuckets = tokens
    print(tokenDenominationBuckets)

    # Optional manual test entry points:
    # plottingTransactionsTest()
    # coinSelectionDistributionTest()
    # Scenario 1: Gaussian payments followed by Gaussian deposits.
    # transactions, deposits, payments = generateDoubleGaussianTransactionsAndPlotThem(plottingIndex=0, noPayments=100000, xFactor=3)
    # Scenario 2: constant deposits with floating-point Dirichlet payments.
    # transactions, deposits, payments = generateTransactions_PaymentsDirichlet_AndPlotThem(plottingIndex=0, noDeposits=100000, xFactor=10, generateDirichletAsFloats=True)
    # Scenario 3: constant deposits with integer multinomial/Dirichlet payments.
    # transactions, deposits, payments = generateTransactions_PaymentsDirichlet_AndPlotThem(plottingIndex=0, noDeposits=100000, xFactor=10, generateDirichletAsFloats=False)
    # simulationTest(tokenDenominationBuckets, transactions)

    # Optional Dirichlet generator checks:
    # r = RandomTransactionGenerator()
    # safeInts = r.generateIntegerDirichletPaymentsViaMultinomial(n=10, sum=2000)
    # print("Generated Dirichlet payments via multinomial distribution:", safeInts)
    # print("Exp. value of Dirichlet payments:", np.mean(safeInts), "+-", np.std(safeInts))
    # safeFloats = r.generateTransactionDirichlet(alpha=1.0, sumValue=2000, sizealpha=10)
    # print("Generated Dirichlet payments via Dirichlet distribution:", safeFloats)
    # print("Exp. value of Dirichlet payments:", np.mean(safeFloats), "+-", np.std(safeFloats))

    runBetaAdjustmentExperimentMatrix(
        tokenDenominationBuckets,
        outputRoot='Simulations/BetaAdjustmentMatrix',
        numSimulations=100,
        noPayments=commandLineArguments.num_iter,
        seed=commandLineArguments.seed,
    )


if __name__ == "__main__":
    main()
