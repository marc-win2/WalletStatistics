import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    def test_cli_defaults_to_one_hundred_payments(self):
        defaultArguments = main.parseCommandLineArguments([])
        self.assertEqual(defaultArguments.num_iter, 100)
        self.assertIsNone(defaultArguments.seed)
        self.assertEqual(
            main.parseCommandLineArguments(["--num_iter", "250"]).num_iter,
            250,
        )
        self.assertEqual(
            main.parseCommandLineArguments(["--seed", "12345"]).seed,
            12345,
        )

    def test_cli_rejects_invalid_payment_counts(self):
        for value in ("0", "-10", "11"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    main.paymentIterationCount(value)

    def test_cli_rejects_negative_seed(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            main.randomSeed("-1")

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
