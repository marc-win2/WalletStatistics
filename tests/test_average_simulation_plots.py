import argparse
import unittest

import numpy as np

from averageSimulationPlots import (
    extendWithValues,
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

    def test_command_line_defaults_match_standard_analysis(self):
        arguments = parseCommandLineArguments([])

        self.assertEqual(arguments.num_runs, 100)
        self.assertEqual(arguments.num_payments, 100000)
        self.assertEqual(arguments.chunk_size, 20000)
        self.assertEqual(arguments.data_path, "./Data")
        self.assertEqual(arguments.save_path, "./DataGlobal")

    def test_command_line_values_are_configurable(self):
        arguments = parseCommandLineArguments(
            [
                "--num_runs", "2",
                "--num_payments", "30",
                "--chunk_size", "7",
                "--data_path", "input",
                "--save_path", "output",
            ]
        )

        self.assertEqual(arguments.num_runs, 2)
        self.assertEqual(arguments.num_payments, 30)
        self.assertEqual(arguments.chunk_size, 7)
        self.assertEqual(arguments.data_path, "input")
        self.assertEqual(arguments.save_path, "output")

    def test_positive_integer_rejects_zero_and_negative_values(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positiveInteger(value)


if __name__ == "__main__":
    unittest.main()
