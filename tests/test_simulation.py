import contextlib
import io
import unittest

from simulation import SimulationHandler
from wallet import Token, Wallet


def make_simulation(**overrides):
    arguments = {
        "tokenDenominationBuckets": [0.01, 1, 10, 100, 10**4, 10**8],
        "beta": 0.1,
        "adjustBetaAfterEachTransaction": False,
        "doEmergRefund": False,
    }
    arguments.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return SimulationHandler(**arguments)


def set_wallet(simulation, values):
    tokens = [Token(value, serialno=index) for index, value in enumerate(values)]
    simulation.highThroughputWallet = Wallet(tokens)
    simulation.tokenNoPerBucket = [0] * len(simulation.tokenBuckets)
    for token in tokens:
        bucketIndex = simulation.getTokensTokenBuckets(token.value)
        simulation.tokenNoPerBucket[bucketIndex] += 1


class SimulationTests(unittest.TestCase):
    def test_initial_wallet_contains_large_funding_token(self):
        simulation = make_simulation()

        self.assertEqual(simulation.highThroughputWallet.getTokenCount(), 1)
        self.assertEqual(simulation.highThroughputWallet.getTotalValue(), 10**7)

    def test_invalid_beta_adjustment_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_simulation(betaAdjustmentMode="unknown")

    def test_seed_reproduces_independent_internal_random_streams(self):
        first = make_simulation(seed=42)
        second = make_simulation(seed=42)

        firstTokenSelectionDraws = first.ownrng.random(4)
        secondTokenSelectionDraws = second.ownrng.random(4)
        firstCoinSelectionDraws = first.coinSelectionDistr.rng.random(4)
        secondCoinSelectionDraws = second.coinSelectionDistr.rng.random(4)

        self.assertTrue(
            (firstTokenSelectionDraws == secondTokenSelectionDraws).all()
        )
        self.assertTrue(
            (firstCoinSelectionDraws == secondCoinSelectionDraws).all()
        )
        self.assertFalse(
            (firstTokenSelectionDraws == firstCoinSelectionDraws).all()
        )

    def test_all_beta_adjustment_formulas(self):
        simulation = make_simulation()
        set_wallet(simulation, [1.0, 2.0, 3.0])

        simulation.adjustBetaMicrocanonically()
        self.assertAlmostEqual(simulation.coinSelectionDistr.beta, 3.0 / 6.0)

        exactBeta = simulation.adjustBetaMicroExact()
        expectedExact = (1.0 / 5.99) + (1.0 / 5.98)
        self.assertAlmostEqual(exactBeta, expectedExact)

        approximateBeta = simulation.adjustBetaMicroApprox()
        self.assertAlmostEqual(approximateBeta, 2.0 / 6.0)

        simulation.betaApproximationFactor = 1000.0
        self.assertAlmostEqual(simulation.adjustBetaMicroApprox(), expectedExact)

    def test_payment_result_counts_selected_input_and_change_token(self):
        simulation = make_simulation()
        set_wallet(simulation, [100.0])

        with contextlib.redirect_stdout(io.StringIO()):
            selectedWallet = simulation.handlePayment(-30.0)

        self.assertEqual(simulation.highThroughputWallet.getTotalValue(), 70.0)
        self.assertEqual(simulation.highThroughputWallet.getTokenCount(), 1)
        self.assertEqual(selectedWallet.getTokenCount(), 2)

    def test_payment_from_empty_wallet_is_rejected_without_modification(self):
        simulation = make_simulation()
        set_wallet(simulation, [])

        with self.assertRaisesRegex(ValueError, "Insufficient wallet funds"):
            simulation.handlePayment(-1.0)

        self.assertEqual(simulation.highThroughputWallet.getTokenCount(), 0)
        self.assertEqual(simulation.highThroughputWallet.getTotalValue(), 0)

    def test_uncovered_payment_is_rejected_without_modifying_wallet(self):
        simulation = make_simulation()
        set_wallet(simulation, [3.0, 7.0])
        originalTokens = list(simulation.highThroughputWallet.tokens)

        with self.assertRaisesRegex(
            ValueError,
            r"payment requires 10\.01, but only 10\.00 is available",
        ):
            simulation.handlePayment(-10.01)

        self.assertEqual(simulation.highThroughputWallet.tokens, originalTokens)
        self.assertEqual(simulation.highThroughputWallet.getTotalValue(), 10.0)

    def test_prolonged_transactions_are_rounded_to_cents(self):
        simulation = make_simulation()
        simulation.prolongTransactionSet([1.234, -2.345])

        self.assertEqual(simulation.transactionSet[-2:], [1.23, -2.35])
        self.assertEqual(simulation.transactionSetSize, 3)


if __name__ == "__main__":
    unittest.main()
