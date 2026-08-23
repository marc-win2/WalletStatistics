import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    def test_cli_defaults_to_one_hundred_payments(self):
        defaultArguments = main.parseCommandLineArguments([])
        self.assertEqual(defaultArguments.num_iter, 100)
        self.assertEqual(defaultArguments.num_runs, 100)
        self.assertEqual(
            defaultArguments.strategies,
            ("boltzmann", "rag_fit", "branch_and_bound"),
        )
        self.assertEqual(
            defaultArguments.beta_output_path,
            "Simulations/BetaAdjustmentMatrix",
        )
        self.assertEqual(
            defaultArguments.strategy_output_path,
            "Simulations/CoinSelectionMatrix",
        )
        self.assertIsNone(defaultArguments.seed)
        self.assertEqual(
            main.parseCommandLineArguments(["--num_iter", "250"]).num_iter,
            250,
        )
        self.assertEqual(
            main.parseCommandLineArguments(["--seed", "12345"]).seed,
            12345,
        )
        self.assertEqual(
            main.parseCommandLineArguments(["--num-runs", "7"]).num_runs,
            7,
        )
        selectedArguments = main.parseCommandLineArguments(
            [
                "--strategies", "rag_fit", "branch_and_bound",
                "--strategy-output-path", "new-results",
            ]
        )
        self.assertEqual(
            selectedArguments.strategies,
            ["rag_fit", "branch_and_bound"],
        )
        self.assertEqual(selectedArguments.strategy_output_path, "new-results")

    def test_cli_rejects_invalid_payment_counts(self):
        for value in ("0", "-10", "11"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    main.paymentIterationCount(value)

    def test_cli_rejects_negative_seed(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            main.randomSeed("-1")

    def test_cli_rejects_invalid_run_counts(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    main.positiveInteger(value)

    def test_run_seeds_pair_transactions_across_beta_modes(self):
        legacySeeds = main.deriveRunSeeds(123, "gaussian", "legacy", 4)
        exactSeeds = main.deriveRunSeeds(
            123,
            "gaussian",
            "microcanonicalExact",
            4,
        )

        self.assertEqual(legacySeeds[0], exactSeeds[0])
        self.assertNotEqual(legacySeeds[1], exactSeeds[1])
        self.assertEqual(
            legacySeeds,
            main.deriveRunSeeds(123, "gaussian", "legacy", 4),
        )
        self.assertNotEqual(
            legacySeeds,
            main.deriveRunSeeds(123, "gaussian", "legacy", 5),
        )
        self.assertEqual(
            main.deriveRunSeeds(None, "gaussian", "legacy", 4),
            (None, None),
        )

    def test_seed_derivation_rejects_unknown_configuration(self):
        with self.assertRaisesRegex(ValueError, "Invalid transaction scenario"):
            main.deriveRunSeeds(123, "unknown", "legacy", 0)
        with self.assertRaisesRegex(ValueError, "Invalid beta adjustment mode"):
            main.deriveRunSeeds(123, "gaussian", "unknown", 0)

    def test_experiment_matrix_has_every_scenario_and_beta_mode(self):
        configurations = main.getBetaAdjustmentExperimentConfigurations()
        combinations = {
            (item["transactionScenario"], item["betaAdjustmentMode"])
            for item in configurations
        }

        self.assertEqual(len(configurations), 6)
        self.assertEqual(
            combinations,
            {
                (scenario, mode)
                for scenario in ("gaussian", "dirichletFloat")
                for mode in (
                    "legacy",
                    "microcanonicalExact",
                    "microcanonicalApprox",
                )
            },
        )

    def test_coin_selection_grid_contains_all_requested_strategies(self):
        configurations = main.getCoinSelectionExperimentConfigurations()

        self.assertEqual(
            [item["coinSelectionStrategy"] for item in configurations],
            ["boltzmann", "rag", "branchAndBound"],
        )
        self.assertEqual(configurations[1]["variant"], "fit")

        workloads = main.getCoinSelectionWorkloadConfigurations()
        self.assertEqual(
            [item["directoryName"] for item in workloads],
            ["Gaussian", "DirichletFloat"],
        )

    @patch("main.runBetaAdjustmentExperiment")
    @patch("main.runBetaAdjustmentExperimentMatrix")
    def test_coin_selection_grid_uses_separate_strategy_directories(
        self, betaMatrix, strategyExperiment
    ):
        betaMatrix.return_value = ["beta-results"]
        strategyExperiment.side_effect = (
            lambda outputRoot, configuration, *_args, **_kwargs:
            os.path.join(outputRoot, configuration["directoryName"])
        )

        directories = main.runCoinSelectionExperimentGrid(
            [0.01, 1.0],
            betaOutputRoot="beta-results",
            strategyOutputRoot="strategy-results",
            numSimulations=2,
            noPayments=10,
            seed=123,
        )

        self.assertEqual(
            directories,
            [
                "beta-results",
                os.path.join("strategy-results", "RAGFit", "Gaussian"),
                os.path.join("strategy-results", "RAGFit", "DirichletFloat"),
                os.path.join("strategy-results", "BranchAndBound", "Gaussian"),
                os.path.join(
                    "strategy-results", "BranchAndBound", "DirichletFloat"
                ),
            ],
        )
        betaMatrix.assert_called_once()
        self.assertEqual(strategyExperiment.call_count, 4)
        self.assertEqual(
            [
                call.kwargs["coinSelectionStrategy"]
                for call in strategyExperiment.call_args_list
            ],
            ["rag", "rag", "branchAndBound", "branchAndBound"],
        )
        self.assertTrue(
            all(
                call.kwargs["adjustBeta"] is False
                for call in strategyExperiment.call_args_list
            )
        )

    @patch("main.runBetaAdjustmentExperiment")
    @patch("main.runBetaAdjustmentExperimentMatrix")
    def test_coin_selection_grid_can_run_only_new_strategies(
        self, betaMatrix, strategyExperiment
    ):
        strategyExperiment.side_effect = (
            lambda outputRoot, configuration, *_args, **_kwargs:
            os.path.join(outputRoot, configuration["directoryName"])
        )

        directories = main.runCoinSelectionExperimentGrid(
            [0.01, 1.0],
            strategyOutputRoot="results",
            strategies=("rag_fit", "branch_and_bound"),
        )

        self.assertEqual(
            directories,
            [
                os.path.join("results", "RAGFit", "Gaussian"),
                os.path.join("results", "RAGFit", "DirichletFloat"),
                os.path.join("results", "BranchAndBound", "Gaussian"),
                os.path.join("results", "BranchAndBound", "DirichletFloat"),
            ],
        )
        betaMatrix.assert_not_called()
        self.assertEqual(strategyExperiment.call_count, 4)

    def test_coin_selection_grid_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            main.runCoinSelectionExperimentGrid(
                [0.01, 1.0], strategies=("unknown",)
            )

    def test_payment_token_counts_exclude_deposits_and_initial_funding(self):
        counts = main.getPaymentTokenCounts(
            transactions=[-10.0, 5.0, -4.0],
            tokenCountPerTransaction=[1, 2, 1, 3],
        )
        self.assertEqual(counts, [2, 3])

    def test_payment_token_counts_can_preserve_zero_value_slots(self):
        counts = main.getPaymentTokenCounts(
            transactions=[-10.0, -0.0, 5.0, -4.0],
            tokenCountPerTransaction=[1, 2, 0, 1, 3],
            includeZeroValuePayments=True,
        )

        self.assertEqual(counts, [2, 0, 3])

    def test_value_writers_use_documented_formats(self):
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            main.writeValues(root / "values.dat", [1.0, 2.0])
            main.writeIndexedValues(root / "indexed.dat", [0, 1], [3.0, 4.0])

            self.assertEqual(
                (root / "values.dat").read_text(),
                "1.0\n2.0\n",
            )
            self.assertEqual(
                (root / "indexed.dat").read_text(),
                "0 3.0\n1 4.0\n",
            )

    @patch("main.plt")
    def test_aggregate_histograms_accept_single_value_files(self, plot):
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            dataDirectory = root / "Data"
            globalDataDirectory = root / "DataGlobal"
            dataDirectory.mkdir()
            globalDataDirectory.mkdir()
            (dataDirectory / "total_transactions.dat").write_text("1.0\n")
            (dataDirectory / "total_token_values_.dat").write_text("2.0\n")

            main.saveAggregateHistograms(dataDirectory, globalDataDirectory)

        self.assertEqual(len(plot.hist.call_args_list), 2)
        self.assertEqual(
            plot.hist.call_args_list[0].args[0].tolist(),
            [1.0],
        )
        self.assertEqual(
            plot.hist.call_args_list[1].args[0].tolist(),
            [2.0],
        )

    @patch("main.generateDoubleGaussianTransactionsAndPlotThem")
    def test_gaussian_scenario_dispatches_to_generator(self, generator):
        generator.return_value = ([1], [2], [-1])

        result = main.generateTransactionScenario("gaussian", 3, 100, "Data")

        self.assertEqual(result, ([1], [2], [-1]))
        generator.assert_called_once_with(
            plottingIndex=3,
            noPayments=100,
            xFactor=3,
            dataDirectory="Data",
            seed=None,
        )


if __name__ == "__main__":
    unittest.main()
