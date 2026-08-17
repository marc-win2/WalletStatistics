"""Compare beta histories for individual runs of an experiment matrix.

Example:
    python3 compareBetaAdjustmentRuns.py \
        --matrix_path Simulations/BetaAdjustmentMatrix_ManyRuns \
        --runs 0 1 2
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BETA_MODES = (
    "legacy",
    "microcanonicalExact",
    "microcanonicalApprox",
)
MODE_LABELS = {
    "legacy": "legacy: n/E",
    "microcanonicalExact": "microcanonical exact",
    "microcanonicalApprox": "microcanonical approximate",
}
MODE_COLORS = {
    "legacy": "tab:blue",
    "microcanonicalExact": "tab:orange",
    "microcanonicalApprox": "tab:green",
}
PAIRWISE_COMPARISONS = (
    ("legacy", "microcanonicalExact"),
    ("legacy", "microcanonicalApprox"),
    ("microcanonicalApprox", "microcanonicalExact"),
)


def parseCommandLineArguments(arguments=None):
    """Parse paths, scenarios, and run indices to compare."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-transaction beta values for individual runs across "
            "the three dynamic beta-adjustment modes."
        )
    )
    parser.add_argument(
        "--matrix_path",
        "--matrix-path",
        default="Simulations/BetaAdjustmentMatrix",
        help="experiment-matrix root directory",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        default=None,
        help="output directory (default: MATRIX_PATH/BetaComparisons)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[0],
        help="run indices to compare (default: 0)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("Gaussian", "DirichletFloat"),
        default=["Gaussian", "DirichletFloat"],
        help="transaction scenarios to compare",
    )
    parser.add_argument(
        "--fail_on_transaction_mismatch",
        "--fail-on-transaction-mismatch",
        action="store_true",
        help="stop instead of warning if beta modes used different transactions",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="output resolution in dots per inch (default: 200)",
    )
    return parser.parse_args(arguments)


def loadBetaHistory(filePath):
    """Load one two-column transaction-index/beta history."""
    values = np.asarray(np.loadtxt(filePath), dtype=float)
    if values.ndim == 1:
        if values.size != 2:
            raise ValueError(f"Expected two columns in {filePath}.")
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected two columns in {filePath}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Found non-finite beta history values in {filePath}.")
    return values[:, 0], values[:, 1]


def loadRunHistories(matrixPath, scenario, runIndex):
    """Load and index-align all beta modes for one scenario and run."""
    histories = {}
    referenceIndices = None
    for mode in BETA_MODES:
        filePath = (
            matrixPath
            / f"{scenario}_{mode}"
            / "Data"
            / f"BetaPerTransaction_{runIndex}.dat"
        )
        if not filePath.is_file():
            raise FileNotFoundError(f"Missing beta history: {filePath}")
        transactionIndices, betaValues = loadBetaHistory(filePath)
        if referenceIndices is None:
            referenceIndices = transactionIndices
        elif not np.array_equal(referenceIndices, transactionIndices):
            raise ValueError(
                f"Transaction indices are not aligned for {scenario} run "
                f"{runIndex}."
            )
        histories[mode] = betaValues
    return referenceIndices, histories


def transactionsMatch(matrixPath, scenario, runIndex):
    """Return whether all beta modes used the same transaction sequence."""
    transactions = []
    for mode in BETA_MODES:
        filePath = (
            matrixPath
            / f"{scenario}_{mode}"
            / "Data"
            / f"transaction{runIndex}.dat"
        )
        if not filePath.is_file():
            return None
        transactions.append(np.atleast_1d(np.loadtxt(filePath)))

    reference = transactions[0]
    return all(
        np.array_equal(reference, candidate)
        for candidate in transactions[1:]
    )


def relativeDifference(leftValues, rightValues):
    """Return absolute relative differences, using NaN for zero references."""
    differences = np.full_like(leftValues, np.nan, dtype=float)
    nonzeroReference = rightValues != 0.0
    differences[nonzeroReference] = (
        np.abs(leftValues[nonzeroReference] - rightValues[nonzeroReference])
        / np.abs(rightValues[nonzeroReference])
    )
    return differences


def setLogScaleWhenUseful(axis, plottedValues):
    """Use a logarithmic scale when at least one positive value exists."""
    if any(np.any(np.asarray(values) > 0.0) for values in plottedValues):
        axis.set_yscale("log")


def finiteStatistic(function, values):
    """Evaluate a statistic on finite values or return NaN."""
    finiteValues = np.asarray(values)[np.isfinite(values)]
    if finiteValues.size == 0:
        return np.nan
    return float(function(finiteValues))


def plotRunComparison(
    matrixPath,
    outputPath,
    scenario,
    runIndex,
    dpi,
    failOnTransactionMismatch=False,
):
    """Create one comparison figure and return its summary rows."""
    transactionIndices, histories = loadRunHistories(
        matrixPath,
        scenario,
        runIndex,
    )
    matchingTransactions = transactionsMatch(matrixPath, scenario, runIndex)
    if matchingTransactions is False:
        message = (
            f"{scenario} run {runIndex} uses different transactions across "
            "beta modes; deviations are descriptive, not a paired comparison."
        )
        if failOnTransactionMismatch:
            raise ValueError(message)
        warnings.warn(message)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for mode in BETA_MODES:
        axes[0].plot(
            transactionIndices,
            histories[mode],
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            linewidth=0.8,
        )
    setLogScaleWhenUseful(axes[0], histories.values())
    axes[0].set_ylabel("beta")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    summaryRows = []
    absoluteDifferences = []
    relativeDifferences = []
    for leftMode, rightMode in PAIRWISE_COMPARISONS:
        label = f"{leftMode} vs {rightMode}"
        absolute = np.abs(histories[leftMode] - histories[rightMode])
        relative = relativeDifference(
            histories[leftMode],
            histories[rightMode],
        )
        absoluteDifferences.append(absolute)
        relativeDifferences.append(relative)
        axes[1].plot(
            transactionIndices,
            absolute,
            label=label,
            linewidth=0.8,
        )
        axes[2].plot(
            transactionIndices,
            100.0 * relative,
            label=label,
            linewidth=0.8,
        )
        summaryRows.append(
            {
                "scenario": scenario,
                "run": runIndex,
                "transactions_match": matchingTransactions,
                "comparison": label,
                "mean_absolute_difference": float(np.mean(absolute)),
                "max_absolute_difference": float(np.max(absolute)),
                "root_mean_square_difference": float(
                    np.sqrt(np.mean(np.square(absolute)))
                ),
                "mean_absolute_relative_difference_percent": (
                    100.0 * finiteStatistic(np.mean, relative)
                ),
                "max_absolute_relative_difference_percent": (
                    100.0 * finiteStatistic(np.max, relative)
                ),
            }
        )

    setLogScaleWhenUseful(axes[1], absoluteDifferences)
    setLogScaleWhenUseful(axes[2], relativeDifferences)
    axes[1].set_ylabel("absolute beta difference")
    axes[2].set_ylabel("relative difference [%]")
    axes[2].set_xlabel("transaction index")
    for axis in axes[1:]:
        axis.legend()
        axis.grid(alpha=0.25)

    if matchingTransactions is True:
        transactionStatus = "identical transactions across beta modes"
    elif matchingTransactions is False:
        transactionStatus = "WARNING: transactions differ across beta modes"
    else:
        transactionStatus = "transaction matching could not be checked"
    figure.suptitle(
        f"{scenario}, run {runIndex}\n{transactionStatus}",
        color="darkred" if matchingTransactions is False else "black",
    )
    figure.tight_layout()
    figure.savefig(
        outputPath / f"{scenario}_run_{runIndex}_beta_comparison.png",
        dpi=dpi,
    )
    plt.close(figure)
    return summaryRows


def writeSummary(filePath, rows):
    """Write pairwise deviation statistics for all generated plots."""
    fieldNames = (
        "scenario",
        "run",
        "transactions_match",
        "comparison",
        "mean_absolute_difference",
        "max_absolute_difference",
        "root_mean_square_difference",
        "mean_absolute_relative_difference_percent",
        "max_absolute_relative_difference_percent",
    )
    with open(filePath, "w", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)


def main(arguments=None):
    """Generate requested per-run beta comparison plots and statistics."""
    commandLineArguments = parseCommandLineArguments(arguments)
    matrixPath = Path(commandLineArguments.matrix_path)
    if not matrixPath.is_dir():
        raise FileNotFoundError(f"Experiment matrix not found: {matrixPath}")
    outputPath = (
        Path(commandLineArguments.output_path)
        if commandLineArguments.output_path is not None
        else matrixPath / "BetaComparisons"
    )
    outputPath.mkdir(parents=True, exist_ok=True)

    summaryRows = []
    for scenario in commandLineArguments.scenarios:
        for runIndex in commandLineArguments.runs:
            if runIndex < 0:
                raise ValueError("Run indices must be non-negative.")
            summaryRows.extend(
                plotRunComparison(
                    matrixPath,
                    outputPath,
                    scenario,
                    runIndex,
                    commandLineArguments.dpi,
                    failOnTransactionMismatch=(
                        commandLineArguments.fail_on_transaction_mismatch
                    ),
                )
            )

    summaryPath = outputPath / "beta_comparison_summary.csv"
    writeSummary(summaryPath, summaryRows)
    plotCount = len(summaryRows) // len(PAIRWISE_COMPARISONS)
    print(f"Saved {plotCount} plots to {outputPath}")
    print(f"Saved summary statistics to {summaryPath}")


if __name__ == "__main__":
    main()
