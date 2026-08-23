import contextlib
import io
import unittest
from unittest.mock import Mock

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

    def test_coin_selection_defaults_to_token_level_boltzmann(self):
        simulation = make_simulation()

        self.assertEqual(simulation.coinSelectionStrategy, "boltzmann")
        self.assertEqual(simulation.samplingMode, "token")
        self.assertFalse(simulation.useBucketsForProbabilityComp)

    def test_invalid_coin_selection_strategy_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid coin-selection strategy",
        ):
            make_simulation(coinSelectionStrategy="unknown")

    def test_invalid_sampling_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid sampling mode"):
            make_simulation(samplingMode="unknown")

    def test_payment_strategies_reject_bucket_legacy_sampling(self):
        for strategy in ("greedy", "branchAndBound", "rag"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(ValueError, "bucketLegacy.*only supported"):
                    make_simulation(
                        coinSelectionStrategy=strategy,
                        samplingMode="bucketLegacy",
                    )

    def test_legacy_bucket_flag_is_rejected_for_payment_strategies(self):
        with self.assertRaisesRegex(ValueError, "bucketLegacy.*only supported"):
            make_simulation(
                coinSelectionStrategy="greedy",
                useBucketsForProbabilityComp=True,
            )

    def test_payment_strategy_configuration_is_validated_at_construction(self):
        invalidConfigurations = (
            {
                "coinSelectionStrategy": "branchAndBound",
                "max_bnb_overshoot": -0.01,
            },
            {"coinSelectionStrategy": "rag", "probability": 1.01},
            {"coinSelectionStrategy": "rag", "target_pool_size": -1},
            {"coinSelectionStrategy": "rag", "variant": "unknown"},
        )
        for configuration in invalidConfigurations:
            with self.subTest(configuration=configuration):
                with self.assertRaises(ValueError):
                    make_simulation(**configuration)

    def test_bucket_legacy_dispatches_to_existing_bucket_selection(self):
        simulation = make_simulation(samplingMode="bucketLegacy")
        selectedToken = simulation.highThroughputWallet.tokens[0]
        simulation.selectTokenBucketFromDistributionThenPickRandom = Mock(
            return_value=(selectedToken, 5, 0.75)
        )

        result = simulation.tokenSelectionProcess(100.0)

        self.assertEqual(result, (selectedToken, 5, 0.75))
        self.assertTrue(simulation.useBucketsForProbabilityComp)
        bucketSelection = (
            simulation.selectTokenBucketFromDistributionThenPickRandom
        )
        bucketSelection.assert_called_once_with()

    def test_legacy_bucket_flag_maps_to_bucket_legacy_sampling(self):
        simulation = make_simulation(useBucketsForProbabilityComp=True)

        self.assertEqual(simulation.samplingMode, "bucketLegacy")
        self.assertTrue(simulation.useBucketsForProbabilityComp)

    def test_bucket_legacy_preserves_legacy_flag_selection_behavior(self):
        namedMode = make_simulation(seed=71, samplingMode="bucketLegacy")
        legacyFlag = make_simulation(
            seed=71,
            useBucketsForProbabilityComp=True,
        )
        values = [0.5, 0.75, 5.0, 50.0]
        set_wallet(namedMode, values)
        set_wallet(legacyFlag, values)

        namedResult = namedMode.tokenSelectionProcess(20.0)
        legacyResult = legacyFlag.tokenSelectionProcess(20.0)

        self.assertEqual(namedResult[0].sno, legacyResult[0].sno)
        self.assertEqual(namedResult[1:], legacyResult[1:])

    def test_distribution_draw_alias_preserves_seeded_token_sampling(self):
        boltzmann = make_simulation(seed=71, coinSelectionStrategy="boltzmann")
        distributionDraw = make_simulation(
            seed=71, coinSelectionStrategy="distributionDraw"
        )
        values = [0.5, 0.75, 5.0, 50.0]
        set_wallet(boltzmann, values)
        set_wallet(distributionDraw, values)

        boltzmannResult = boltzmann.tokenSelectionProcess(20.0)
        distributionDrawResult = distributionDraw.tokenSelectionProcess(20.0)

        self.assertEqual(boltzmannResult[0].sno, distributionDrawResult[0].sno)
        self.assertEqual(boltzmannResult[1:], distributionDrawResult[1:])

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

        simulation.adjustBetaMicrocanonicalLegacy()
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

    def test_greedy_payment_strategy_applies_complete_plan_and_change(self):
        simulation = make_simulation(coinSelectionStrategy="greedy")
        set_wallet(simulation, [10.0, 3.0])

        selected = simulation.handlePayment(-11.0)

        self.assertEqual([token.value for token in selected.tokens], [10.0, 3.0, 2.0])
        self.assertEqual([token.value for token in simulation.highThroughputWallet.tokens], [2.0])
        self.assertEqual(simulation.lastSelectionPlan.strategy, "greedy")
        self.assertEqual(simulation.lastSelectionPlan.change, 2.0)
        self.assertEqual(simulation.tokenNoPerBucket, [0, 0, 1, 0, 0, 0])
        self.assertEqual(
            simulation.highThroughputWallet.getTotalValue() + 11.0,
            13.0,
        )

    def test_payment_plan_creates_one_change_token_and_updates_beta_per_input(self):
        simulation = make_simulation(
            coinSelectionStrategy="greedy", adjustBetaAfterEachTransaction=True
        )
        set_wallet(simulation, [10.0, 3.0])
        simulation.adjustBetaDynamically = Mock()

        selected = simulation.handlePayment(-11.0)

        self.assertEqual([token.value for token in selected.tokens], [10.0, 3.0, 2.0])
        self.assertEqual(sum(token.value == 2.0 for token in selected.tokens), 1)
        self.assertEqual(simulation.adjustBetaDynamically.call_count, 2)

    def test_branch_and_bound_overshoot_creates_change_and_exposes_plan(self):
        simulation = make_simulation(
            coinSelectionStrategy="branchAndBound", max_bnb_overshoot=3.0
        )
        set_wallet(simulation, [10.0])

        selected = simulation.handlePayment(-7.0)

        self.assertEqual([token.value for token in selected.tokens], [10.0, 3.0])
        self.assertEqual(simulation.highThroughputWallet.getTotalValue(), 3.0)
        self.assertEqual(simulation.lastSelectionPlan.change, 3.0)
        self.assertFalse(simulation.lastSelectionPlan.bnb_fallback_used)

    def test_branch_and_bound_fallback_is_observable(self):
        simulation = make_simulation(coinSelectionStrategy="branchAndBound")
        set_wallet(simulation, [1.0] * 2001)

        simulation.handlePayment(-1.0)

        self.assertTrue(simulation.lastSelectionPlan.bnb_fallback_used)
        self.assertEqual(
            simulation.lastSelectionPlan.strategy, "branch_and_bound_fallback"
        )

    def test_seeded_rag_payment_is_reproducible(self):
        first = make_simulation(
            coinSelectionStrategy="rag", probability=0.4, seed=17
        )
        second = make_simulation(
            coinSelectionStrategy="rag", probability=0.4, seed=17
        )
        set_wallet(first, [9.0, 7.0, 5.0, 3.0])
        set_wallet(second, [9.0, 7.0, 5.0, 3.0])

        firstSelected = first.handlePayment(-10.0)
        secondSelected = second.handlePayment(-10.0)

        self.assertEqual(
            [token.sno for token in firstSelected.tokens],
            [token.sno for token in secondSelected.tokens],
        )
        self.assertEqual(first.lastSelectionPlan.selected_total, second.lastSelectionPlan.selected_total)
        self.assertEqual(first.lastSelectionPlan.change, second.lastSelectionPlan.change)

    def test_rag_defaults_to_fit_variant(self):
        simulation = make_simulation(coinSelectionStrategy="rag", probability=1.0)
        set_wallet(simulation, [12.0, 6.0, 4.0])

        simulation.handlePayment(-10.0)

        self.assertEqual(
            [token.value for token in simulation.lastSelectionPlan.inputs],
            [6.0, 4.0],
        )
        self.assertEqual(simulation.lastSelectionPlan.strategy, "rag_fit")

    def test_payment_level_strategy_insufficient_funds_is_non_mutating(self):
        simulation = make_simulation(coinSelectionStrategy="rag", probability=1.0)
        set_wallet(simulation, [3.0, 2.0])
        before = [(token.value, token.sno) for token in simulation.highThroughputWallet.tokens]

        with self.assertRaisesRegex(ValueError, "Insufficient wallet funds"):
            simulation.handlePayment(-6.0)

        self.assertEqual(
            [(token.value, token.sno) for token in simulation.highThroughputWallet.tokens],
            before,
        )
        self.assertIsNone(simulation.lastSelectionPlan)

    def test_failed_payment_planning_preserves_the_last_successful_plan(self):
        simulation = make_simulation(coinSelectionStrategy="greedy")
        set_wallet(simulation, [10.0])
        simulation.handlePayment(-7.0)
        successfulPlan = simulation.lastSelectionPlan
        before = [(token.value, token.sno) for token in simulation.highThroughputWallet.tokens]

        with self.assertRaisesRegex(ValueError, "Insufficient wallet funds"):
            simulation.handlePayment(-4.0)

        self.assertIs(simulation.lastSelectionPlan, successfulPlan)
        self.assertEqual(
            [(token.value, token.sno) for token in simulation.highThroughputWallet.tokens],
            before,
        )

    def test_prolonged_transactions_are_rounded_to_cents(self):
        simulation = make_simulation()
        simulation.prolongTransactionSet([1.234, -2.345])

        self.assertEqual(simulation.transactionSet[-2:], [1.23, -2.35])
        self.assertEqual(simulation.transactionSetSize, 3)


if __name__ == "__main__":
    unittest.main()
