import tempfile
import unittest
from pathlib import Path

import numpy as np

import compareBetaAdjustmentRuns as comparison


def write_run(root, scenario, run_index, transactions_by_mode=None):
    if transactions_by_mode is None:
        transactions_by_mode = {
            mode: [-2.0, 1.0]
            for mode in comparison.BETA_MODES
        }
    for mode_index, mode in enumerate(comparison.BETA_MODES):
        data_path = root / f"{scenario}_{mode}" / "Data"
        data_path.mkdir(parents=True)
        beta_values = [1.0, 0.5 + 0.1 * mode_index, 0.25]
        beta_rows = "".join(
            f"{index} {value}\n"
            for index, value in enumerate(beta_values)
        )
        (data_path / f"BetaPerTransaction_{run_index}.dat").write_text(
            beta_rows
        )
        transaction_rows = "".join(
            f"{value}\n" for value in transactions_by_mode[mode]
        )
        (data_path / f"transaction{run_index}.dat").write_text(
            transaction_rows
        )


class CompareBetaAdjustmentRunsTests(unittest.TestCase):
    def test_command_line_defaults_to_run_zero_and_both_scenarios(self):
        arguments = comparison.parseCommandLineArguments([])

        self.assertEqual(arguments.runs, [0])
        self.assertEqual(arguments.scenarios, ["Gaussian", "DirichletFloat"])
        self.assertFalse(arguments.fail_on_transaction_mismatch)

    def test_relative_difference_ignores_zero_reference_values(self):
        result = comparison.relativeDifference(
            np.array([2.0, 3.0]),
            np.array([1.0, 0.0]),
        )

        self.assertEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))

    def test_histories_are_loaded_with_aligned_indices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_run(root, "Gaussian", 2)

            indices, histories = comparison.loadRunHistories(
                root,
                "Gaussian",
                2,
            )

            np.testing.assert_array_equal(indices, [0.0, 1.0, 2.0])
            self.assertEqual(set(histories), set(comparison.BETA_MODES))
            self.assertTrue(comparison.transactionsMatch(root, "Gaussian", 2))

    def test_transaction_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            transactions = {
                mode: [-2.0, float(mode_index)]
                for mode_index, mode in enumerate(comparison.BETA_MODES)
            }
            write_run(root, "DirichletFloat", 0, transactions)

            self.assertFalse(
                comparison.transactionsMatch(root, "DirichletFloat", 0)
            )

    def test_plot_and_summary_are_created_for_matching_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "output"
            output_path.mkdir()
            write_run(root, "Gaussian", 1)

            rows = comparison.plotRunComparison(
                root,
                output_path,
                "Gaussian",
                1,
                dpi=50,
                failOnTransactionMismatch=True,
            )
            comparison.writeSummary(output_path / "summary.csv", rows)

            self.assertEqual(len(rows), 3)
            self.assertTrue(
                (output_path / "Gaussian_run_1_beta_comparison.png").is_file()
            )
            self.assertTrue((output_path / "summary.csv").is_file())
            self.assertTrue(all(row["transactions_match"] for row in rows))


if __name__ == "__main__":
    unittest.main()
