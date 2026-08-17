import contextlib
import io
import unittest

import numpy as np

from coinselection import BoltzmannDistribution, CoinSelectionDistribution
from wallet import Token


def make_distribution(beta=0.1, mode="canonical"):
    with contextlib.redirect_stdout(io.StringIO()):
        return CoinSelectionDistribution(beta, [1, 10, 100], mode)


class CoinSelectionTests(unittest.TestCase):
    def test_boltzmann_distribution_uses_energy_beta_and_mu(self):
        self.assertAlmostEqual(
            BoltzmannDistribution(3.0, 0.5, mu=1.0),
            np.exp(-1.0),
        )

    def test_discrete_distribution_normalizes_token_weights(self):
        distribution = make_distribution(beta=0.1)
        tokens = [Token(1.0, 1), Token(2.0, 2)]

        probabilities, cumulativeWeights = distribution.compDistributionDiscrSet(
            tokens,
            transactionValue=2.0,
        )
        expectedWeights = np.exp(-0.1 * np.array([1.0, 2.0]))

        np.testing.assert_allclose(
            probabilities,
            expectedWeights / expectedWeights.sum(),
        )
        np.testing.assert_allclose(cumulativeWeights, np.cumsum(expectedWeights))

    def test_discrete_denominator_uses_token_values(self):
        distribution = make_distribution(beta=0.1)
        tokens = [Token(1.0, 1), Token(2.0, 2)]

        denominator = distribution.compDenominatorDiscDistribution(tokens)

        self.assertAlmostEqual(
            denominator,
            np.exp(-0.1) + np.exp(-0.2),
        )

    def test_empty_token_set_returns_two_empty_results(self):
        distribution = make_distribution()
        with contextlib.redirect_stdout(io.StringIO()):
            result = distribution.compDistributionDiscrSet([], 1.0)

        self.assertEqual(result, ([], []))

    def test_underflow_falls_back_to_uniform_token_probabilities(self):
        distribution = make_distribution(beta=1000.0)
        tokens = [Token(1.0, 1), Token(2.0, 2)]

        probabilities, cumulativeWeights = distribution.compDistributionDiscrSet(
            tokens,
            transactionValue=2.0,
        )

        self.assertEqual(probabilities, [0.5, 0.5])
        self.assertEqual(cumulativeWeights, [0.5, 1.0])

    def test_bucket_probabilities_ignore_empty_buckets_and_use_current_beta(self):
        distribution = make_distribution(beta=0.1)
        distribution.setBeta(0.2)

        probabilities, cumulativeWeights = (
            distribution.returnBucketProbabilitiesForFixedWalletState([1, 0, 1])
        )

        expected = [np.exp(-0.2), 0.0, np.exp(-20.0)]
        np.testing.assert_allclose(probabilities, expected)
        np.testing.assert_allclose(cumulativeWeights, np.cumsum(expected))

    def test_uniform_mode_sets_beta_to_zero(self):
        distribution = make_distribution(beta=1.0)
        distribution.setUniform()

        self.assertEqual(distribution.mode, "uniform")
        self.assertEqual(distribution.beta, 0.0)
        self.assertEqual(distribution.compProbability(10.0), 0.1)

    def test_grandcanonical_constructor_initializes_chemical_potentials(self):
        distribution = make_distribution(beta=0.1, mode="grandcanonical")

        self.assertEqual(distribution.mode, "grandcanonical")
        self.assertEqual(distribution.muArray, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            distribution.preComputedProbabilities,
            np.exp(-0.1 * np.array([1.0, 10.0, 100.0])),
        )

    def test_grandcanonical_scalar_and_array_setters_remain_available(self):
        distribution = make_distribution()
        distribution.setGrandCanonical(2.0)
        self.assertEqual(distribution.muArray, [2.0, 2.0, 2.0])

        distribution.setGrandCanonicalArray([1.0, 2.0, 3.0])
        self.assertEqual(distribution.muArray, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
