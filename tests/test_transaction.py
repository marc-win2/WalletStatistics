import contextlib
import io
import unittest

import numpy as np

from transaction import (
    RandomTransactionGenerator,
    generateGaussianFloats,
    generateUniformFloats,
    initializeRandomNumGenerator,
)


class TransactionTests(unittest.TestCase):
    def test_seeded_generators_are_reproducible(self):
        first = initializeRandomNumGenerator(123)
        second = initializeRandomNumGenerator(123)

        np.testing.assert_allclose(
            generateGaussianFloats(first, 5),
            generateGaussianFloats(second, 5),
        )

    def test_uniform_values_stay_inside_requested_interval(self):
        rng = initializeRandomNumGenerator(7)
        values = generateUniformFloats(rng, 100, -2.0, 3.0)

        self.assertTrue(np.all(values >= -2.0))
        self.assertTrue(np.all(values < 3.0))

    def test_gaussian_transactions_are_rounded_and_bounded(self):
        generator = RandomTransactionGenerator(seed=4)
        generator.maxAbsTransactionValue = 50
        values = generator.generateNTransactionsGaussian(20, stdDev=100, mean=0)

        self.assertTrue(all(abs(value) <= 50 for value in values))
        self.assertTrue(all(value == round(value, 2) for value in values))

    def test_dirichlet_transaction_preserves_total_value(self):
        generator = RandomTransactionGenerator(seed=5)
        values = generator.generateTransactionDirichlet(
            alpha=1.0,
            sumValue=2000,
            sizealpha=10,
        )

        self.assertEqual(len(values), 10)
        self.assertAlmostEqual(float(np.sum(values)), 2000.0, places=2)
        self.assertTrue(all(value == round(float(value), 2) for value in values))

    def test_multinomial_generator_returns_requested_number_of_parts(self):
        generator = RandomTransactionGenerator(seed=9)
        with contextlib.redirect_stdout(io.StringIO()):
            values = generator.generateIntegerDirichletPaymentsViaMultinomial(
                n=10,
                sum=2000,
            )

        self.assertEqual(len(values), 10)
        self.assertEqual(int(np.sum(values)), 2000)


if __name__ == "__main__":
    unittest.main()
