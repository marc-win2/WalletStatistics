import unittest

import numpy as np

from averageSimulationPlots import extendWithValues, meanAndStd


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


if __name__ == "__main__":
    unittest.main()
