import argparse
import os
import tempfile
import unittest

import numpy as np

from averageSimulationPlots import (
    inferPaymentCount,
    extendWithValues,
    loadDataValues,
    meanAndStd,
    parseCommandLineArguments,
    positiveInteger,
)


class AverageSimulationPlotsTests(unittest.TestCase):
    def test_mean_and_standard_deviation_use_population_statistics(self):
        mean, standardDeviation = meanAndStd([1.0, 3.0])

        self.assertEqual(mean, 2.0)
        self.assertEqual(standardDeviation, 1.0)

    def test_extend_with_values_accepts_arrays_and_scalars(self):
        values = []
        extendWithValues(values, np.array([1.0, 2.0]))
        extendWithValues(values, np.array(3.0))

        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_data_loader_accepts_empty_and_single_value_files(self):
        with tempfile.TemporaryDirectory() as dataPath:
            emptyPath = os.path.join(dataPath, "empty.dat")
            singlePath = os.path.join(dataPath, "single.dat")
            with open(emptyPath, "w", encoding="utf-8"):
                pass
            with open(singlePath, "w", encoding="utf-8") as dataFile:
                dataFile.write("3.5\n")

            self.assertEqual(loadDataValues(emptyPath).tolist(), [])
            self.assertEqual(loadDataValues(singlePath).tolist(), [3.5])

    def test_command_line_defaults_match_standard_analysis(self):
        arguments = parseCommandLineArguments([])

        self.assertEqual(arguments.num_runs, 100)
        self.assertEqual(arguments.chunk_size, 20000)
        self.assertEqual(arguments.data_path, "./Data")
        self.assertEqual(arguments.save_path, "./DataGlobal")

    def test_command_line_values_are_configurable(self):
        arguments = parseCommandLineArguments(
            [
                "--num_runs", "2",
                "--chunk_size", "7",
                "--data_path", "input",
                "--save_path", "output",
            ]
        )

        self.assertEqual(arguments.num_runs, 2)
        self.assertEqual(arguments.chunk_size, 7)
        self.assertEqual(arguments.data_path, "input")
        self.assertEqual(arguments.save_path, "output")

    def test_positive_integer_rejects_zero_and_negative_values(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positiveInteger(value)

    def test_payment_count_is_inferred_from_all_requested_runs(self):
        with tempfile.TemporaryDirectory() as dataPath:
            for simulationIndex in range(2):
                filePath = os.path.join(
                    dataPath, f"payment_token_count_{simulationIndex}.dat"
                )
                with open(filePath, "w", encoding="utf-8") as dataFile:
                    dataFile.write("1\n2\n3\n")

            self.assertEqual(inferPaymentCount(dataPath, 2), 3)

    def test_different_payment_counts_use_the_shortest_common_history(self):
        with tempfile.TemporaryDirectory() as dataPath:
            for simulationIndex, values in enumerate(("1\n2\n", "1\n")):
                filePath = os.path.join(
                    dataPath, f"payment_token_count_{simulationIndex}.dat"
                )
                with open(filePath, "w", encoding="utf-8") as dataFile:
                    dataFile.write(values)

            self.assertEqual(inferPaymentCount(dataPath, 2), 1)


if __name__ == "__main__":
    unittest.main()
