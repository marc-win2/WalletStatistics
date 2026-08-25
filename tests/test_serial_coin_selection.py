import unittest
from unittest.mock import patch

from serial_coin_selection import (
    BranchAndBoundStrategy,
    GreedyStrategy,
    InsufficientFundsError,
    RagVariant,
    RandomizedAdaptiveGreedyStrategy,
    plan_branch_and_bound_selection,
    plan_greedy_selection,
    plan_randomized_adaptive_greedy_selection,
)
from wallet import Token, Wallet


class GreedyStrategyTests(unittest.TestCase):
    def test_exact_single_coin_match_retains_original_token(self):
        token = Token(5.00, serialno=17)

        plan = plan_greedy_selection([token, Token(2.00, serialno=18)], 5.00)

        self.assertEqual(plan.inputs, (token,))
        self.assertEqual(plan.selected_total, 5.00)
        self.assertEqual(plan.payment_amount, 5.00)
        self.assertEqual(plan.change, 0.00)
        self.assertFalse(plan.changeless)
        self.assertEqual(plan.strategy, "greedy")
        self.assertFalse(plan.bnb_fallback_used)
        self.assertFalse(plan.used_bnb_fallback)

    def test_exact_multi_coin_match(self):
        plan = plan_greedy_selection(
            [Token(5.00, 1), Token(3.00, 2), Token(2.00, 3)], 10.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 2, 3])
        self.assertEqual(plan.selected_total_cents, 1000)
        self.assertEqual(plan.change, 0.00)

    def test_prefers_largest_token_that_fits_remaining_payment(self):
        plan = GreedyStrategy().select(
            [Token(10.00, 1), Token(8.00, 2), Token(3.00, 3)], 11.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 3])
        self.assertEqual(plan.selected_total, 13.00)
        self.assertEqual(plan.change, 2.00)

    def test_uses_smallest_token_for_unavoidable_overshoot(self):
        plan = plan_greedy_selection(
            [Token(10.00, 1), Token(7.00, 2), Token(3.00, 3)], 11.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 3])
        self.assertEqual(plan.change, 2.00)

    def test_insufficient_funds_is_pure(self):
        tokens = [Token(6.00, 1), Token(4.00, 2)]
        wallet = Wallet(tokens)
        original_wallet_tokens = list(wallet.tokens)

        with self.assertRaisesRegex(
            InsufficientFundsError,
            r"payment requires 10\.01, but only 10\.00 is available",
        ):
            plan_greedy_selection(wallet.tokens, 10.01)

        self.assertEqual(wallet.tokens, original_wallet_tokens)
        self.assertEqual([token.sno for token in wallet.tokens], [1, 2])

    def test_successful_selection_is_pure_and_retains_token_identity(self):
        tokens = [Token(10.00, 1), Token(3.00, 2), Token(8.00, 3)]
        original_tokens = list(tokens)

        plan = plan_greedy_selection(tokens, 11.00)

        self.assertEqual(tokens, original_tokens)
        self.assertIs(plan.inputs[0], tokens[0])
        self.assertIs(plan.inputs[1], tokens[1])

    def test_rejects_non_finite_or_sub_cent_amounts(self):
        for amount in (float("nan"), float("inf"), 1.001):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    plan_greedy_selection([Token(2.00, 1)], amount)

    def test_rejects_non_positive_payment(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            plan_greedy_selection([Token(1.00, 1)], 0.00)


class BranchAndBoundStrategyTests(unittest.TestCase):
    def test_finds_exact_subset(self):
        plan = plan_branch_and_bound_selection(
            [Token(7.00, 1), Token(6.00, 2), Token(4.00, 3)], 10.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 3])
        self.assertEqual(plan.selected_total, 10.00)
        self.assertEqual(plan.change, 0.00)
        self.assertFalse(plan.changeless)
        self.assertEqual(plan.strategy, "branch_and_bound")
        self.assertFalse(plan.bnb_fallback_used)

    def test_keeps_minimum_overshoot_within_explicit_cap(self):
        plan = plan_branch_and_bound_selection(
            [Token(14.00, 1), Token(12.00, 2), Token(6.00, 3)],
            10.00,
            max_bnb_overshoot=5.00,
        )

        self.assertEqual([token.sno for token in plan.inputs], [2])
        self.assertEqual(plan.selected_total, 12.00)
        self.assertEqual(plan.change, 2.00)
        self.assertFalse(plan.bnb_fallback_used)

    def test_zero_overshoot_cap_uses_fallback_when_no_exact_candidate_exists(self):
        plan = plan_branch_and_bound_selection(
            [Token(10.00, 1)], 9.00, max_bnb_overshoot=0.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [1])
        self.assertEqual(plan.change, 1.00)
        self.assertEqual(plan.strategy, "branch_and_bound_fallback")
        self.assertTrue(plan.bnb_fallback_used)
        self.assertTrue(plan.used_bnb_fallback)

    def test_default_twenty_percent_cap_accepts_inclusive_boundary(self):
        plan = plan_branch_and_bound_selection(
            [Token(14.00, 1), Token(12.00, 2), Token(6.00, 3)], 10.00
        )

        self.assertEqual([token.sno for token in plan.inputs], [2])
        self.assertEqual(plan.selected_total, 12.00)
        self.assertEqual(plan.change, 2.00)
        self.assertEqual(plan.strategy, "branch_and_bound")
        self.assertFalse(plan.bnb_fallback_used)

    def test_default_twenty_percent_cap_falls_back_above_boundary(self):
        token = Token(12.01, 1)

        plan = plan_branch_and_bound_selection([token], 10.00)

        self.assertEqual(plan.inputs, (token,))
        self.assertEqual(plan.change, 2.01)
        self.assertEqual(plan.strategy, "branch_and_bound_fallback")
        self.assertTrue(plan.bnb_fallback_used)

    def test_explicit_overshoot_cap_causes_fallback_outside_bound(self):
        token = Token(12.00, 1)
        plan = plan_branch_and_bound_selection(
            [token], 10.00, max_bnb_overshoot=1.00
        )

        self.assertEqual(plan.inputs, (token,))
        self.assertEqual(plan.selected_total, 12.00)
        self.assertEqual(plan.change, 2.00)
        self.assertEqual(plan.strategy, "branch_and_bound_fallback")
        self.assertTrue(plan.bnb_fallback_used)

    def test_more_than_one_hundred_eighty_tokens_uses_fallback(self):
        tokens = [Token(10.00, 1)] + [
            Token(1.00, serialno) for serialno in range(2, 182)
        ]

        plan = plan_branch_and_bound_selection(tokens, 9.00)

        self.assertEqual(plan.inputs, tuple(tokens[1:10]))
        self.assertEqual(plan.change, 0.00)
        self.assertEqual(plan.strategy, "branch_and_bound_fallback")
        self.assertTrue(plan.bnb_fallback_used)

    def test_fallback_uses_smallest_input_only_for_unavoidable_overshoot(self):
        tokens = [Token(12.00, 1), Token(11.00, 2), Token(4.00, 3)]

        with patch.object(BranchAndBoundStrategy, "MAX_SEARCH_ATTEMPTS", 0):
            plan = plan_branch_and_bound_selection(tokens, 10.00)

        self.assertEqual(plan.inputs, (tokens[2], tokens[1]))
        self.assertEqual(plan.selected_total, 15.00)
        self.assertEqual(plan.change, 5.00)
        self.assertTrue(plan.bnb_fallback_used)

    def test_exactly_one_hundred_eighty_tokens_remains_eligible_for_bnb(self):
        tokens = [Token(10.00, 1)] + [Token(1.00, serialno) for serialno in range(2, 181)]

        plan = plan_branch_and_bound_selection(tokens, 10.00)

        self.assertEqual(plan.inputs, (tokens[0],))
        self.assertFalse(plan.bnb_fallback_used)

    def test_search_budget_exhaustion_uses_fallback(self):
        with patch.object(BranchAndBoundStrategy, "MAX_SEARCH_ATTEMPTS", 0):
            plan = plan_branch_and_bound_selection([Token(10.00, 1)], 9.00)

        self.assertEqual(plan.strategy, "branch_and_bound_fallback")
        self.assertTrue(plan.bnb_fallback_used)

    def test_duplicate_denominations_preserve_individual_token_identity(self):
        first_six = Token(6.00, 1)
        second_six = Token(6.00, 2)
        first_four = Token(4.00, 3)
        second_four = Token(4.00, 4)

        plan = plan_branch_and_bound_selection(
            [first_six, second_six, first_four, second_four], 10.00
        )

        self.assertIs(plan.inputs[0], first_six)
        self.assertIs(plan.inputs[1], first_four)

    def test_insufficient_funds_is_pure(self):
        tokens = [Token(6.00, 1), Token(4.00, 2)]
        original_tokens = list(tokens)

        with self.assertRaises(InsufficientFundsError):
            plan_branch_and_bound_selection(tokens, 10.01)

        self.assertEqual(tokens, original_tokens)

    def test_successful_overshoot_has_ordinary_change_and_conserves_value(self):
        tokens = [Token(11.00, 1)]
        plan = plan_branch_and_bound_selection(
            tokens, 10.00, max_bnb_overshoot=1.00
        )

        self.assertEqual(plan.selected_total, 11.00)
        self.assertEqual(plan.change, 1.00)
        self.assertFalse(plan.changeless)
        self.assertEqual(sum(token.value for token in tokens), plan.payment_amount + plan.change)

    def test_accepts_the_inclusive_upper_boundary(self):
        plan = plan_branch_and_bound_selection(
            [Token(12.00, 1)], 10.00, max_bnb_overshoot=2.00
        )

        self.assertEqual(plan.selected_total, 12.00)
        self.assertEqual(plan.change, 2.00)
        self.assertFalse(plan.bnb_fallback_used)

    def test_equal_value_ties_are_deterministic_and_inclusion_first(self):
        plan = plan_branch_and_bound_selection(
            [Token(4.00, 4), Token(6.00, 2), Token(4.00, 3), Token(6.00, 1)],
            10.00,
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 4])


class SequenceRng:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self.values)


class ConstantRng:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


class RandomizedAdaptiveGreedyTests(unittest.TestCase):
    def test_defaults_probability_to_one_half(self):
        strategy = RandomizedAdaptiveGreedyStrategy()

        self.assertEqual(strategy.probability, 0.5)

    def test_rejects_invalid_probability(self):
        for probability in (float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    RandomizedAdaptiveGreedyStrategy(probability)

    def test_probability_one_default_fit_matches_greedy(self):
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(12.00, 1), Token(6.00, 2), Token(4.00, 3)],
            10.00,
            probability=1.0,
            rng=ConstantRng(0.0),
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 3])
        self.assertEqual(plan.change, 0.00)
        self.assertEqual(plan.strategy, "rag_fit")

    def test_default_target_pool_size_adapts_above_twenty_tokens(self):
        tokens = [Token(6.00, 1), Token(4.00, 2), Token(0.50, 3)]
        tokens.extend(Token(0.01, serialno) for serialno in range(4, 22))

        plan = plan_randomized_adaptive_greedy_selection(
            tokens,
            10.00,
            probability=1.0,
            rng=ConstantRng(0.0),
        )

        # With 21 tokens, the default target is floor(10.00 * 21 / 20) = 10.50.
        self.assertEqual([token.sno for token in plan.inputs], [1, 2, 3])
        self.assertEqual(plan.selected_total, 10.50)
        self.assertEqual(plan.change, 0.50)

    def test_probability_zero_uses_largest_first_safety_top_up(self):
        rng = ConstantRng(0.5)
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(8.00, 1), Token(5.00, 2), Token(3.00, 3)],
            10.00,
            probability=0.0,
            variant=RagVariant.LargestFirst,
            rng=rng,
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 2])
        self.assertEqual(plan.change, 3.00)
        self.assertEqual(rng.calls, 3 * RandomizedAdaptiveGreedyStrategy.MAX_ATTEMPTS)

    def test_target_pool_size_boundary_disables_adaptive_scaling(self):
        for target_pool_size in (None, 0, 3, 4):
            with self.subTest(target_pool_size=target_pool_size):
                plan = plan_randomized_adaptive_greedy_selection(
                    [Token(12.00, 1), Token(6.00, 2), Token(4.00, 3)],
                    10.00,
                    probability=1.0,
                    target_pool_size=target_pool_size,
                    variant=RagVariant.SmallestFirstConsolidate,
                    rng=ConstantRng(0.0),
                )

                self.assertEqual([token.sno for token in plan.inputs], [1])
                self.assertEqual(plan.change, 2.00)

    def test_adaptive_smallest_first_consolidates_and_change_uses_payment(self):
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(12.00, 1), Token(6.00, 2), Token(4.00, 3)],
            10.00,
            probability=1.0,
            target_pool_size=2,
            variant=RagVariant.SmallestFirstConsolidate,
            rng=ConstantRng(0.0),
        )

        # Adaptive target is floor(10 * 3 / 2) = 15.  Ascending scans take
        # 4, then 6, then 12; change is relative to the actual payment, 10.
        self.assertEqual([token.sno for token in plan.inputs], [3, 2, 1])
        self.assertEqual(plan.selected_total, 22.00)
        self.assertEqual(plan.change, 12.00)
        self.assertFalse(plan.changeless)

    def test_adaptive_target_matches_rust_float_truncation(self):
        tokens = [Token(0.01, serialno) for serialno in range(1, 62)]
        plan = plan_randomized_adaptive_greedy_selection(
            tokens,
            0.07,
            probability=1.0,
            target_pool_size=7,
            variant=RagVariant.SmallestFirstConsolidate,
            rng=ConstantRng(0.0),
        )

        # Rust computes int(7.0 * (61.0 / 7.0)) == 60, not exact rational 61.
        self.assertEqual(len(plan.inputs), 60)
        self.assertEqual(plan.selected_total, 0.60)
        self.assertEqual(plan.change, 0.53)

    def test_largest_first_and_smallest_first_non_adaptive_are_unfiltered(self):
        tokens = [Token(12.00, 1), Token(6.00, 2), Token(4.00, 3)]
        for variant in (RagVariant.LargestFirst, RagVariant.SmallestFirstConsolidate):
            with self.subTest(variant=variant):
                plan = plan_randomized_adaptive_greedy_selection(
                    tokens,
                    10.00,
                    probability=1.0,
                    variant=variant,
                    rng=ConstantRng(0.0),
                )
                self.assertEqual([token.sno for token in plan.inputs], [1])
                self.assertEqual(plan.change, 2.00)

    def test_fit_stops_after_payment_when_no_token_fits_adaptive_remainder(self):
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(20.00, 1), Token(6.00, 2), Token(4.00, 3)],
            10.00,
            probability=1.0,
            target_pool_size=2,
            variant=RagVariant.Fit,
            rng=ConstantRng(0.0),
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 3])
        self.assertEqual(plan.selected_total, 10.00)
        self.assertEqual(plan.change, 0.00)

    def test_fit_uses_smallest_without_random_draw_when_nothing_fits(self):
        rng = ConstantRng(0.0)
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(12.00, 1), Token(7.00, 2), Token(6.00, 3)],
            8.00,
            probability=1.0,
            variant=RagVariant.Fit,
            rng=rng,
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 3])
        self.assertEqual(plan.change, 5.00)
        self.assertEqual(rng.calls, 1)

    def test_fit_retries_a_failed_fitting_scan(self):
        rng = SequenceRng([0.9, 0.9, 0.1, 0.1])
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(6.00, 1), Token(4.00, 2)],
            10.00,
            probability=0.5,
            variant=RagVariant.Fit,
            rng=rng,
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 2])
        self.assertEqual(rng.calls, 4)

    def test_adaptive_largest_first_remains_descending(self):
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(12.00, 1), Token(6.00, 2), Token(4.00, 3)],
            10.00,
            probability=1.0,
            target_pool_size=2,
            variant=RagVariant.LargestFirst,
            rng=ConstantRng(0.0),
        )

        self.assertEqual([token.sno for token in plan.inputs], [1, 2])
        self.assertEqual(plan.selected_total, 18.00)

    def test_probability_is_checked_independently_for_duplicate_tokens(self):
        rng = SequenceRng([0.9, 0.1])
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(5.00, 1), Token(5.00, 2), Token(3.00, 3)],
            3.00,
            probability=0.5,
            variant=RagVariant.LargestFirst,
            rng=rng,
        )

        self.assertEqual([token.sno for token in plan.inputs], [2])
        self.assertEqual(plan.change, 2.00)
        self.assertEqual(rng.calls, 2)

    def test_injected_rng_makes_selection_deterministic(self):
        rng = SequenceRng([0.9, 0.1, 0.1])
        plan = plan_randomized_adaptive_greedy_selection(
            [Token(9.00, 1), Token(6.00, 2), Token(4.00, 3)],
            10.00,
            probability=0.5,
            variant=RagVariant.LargestFirst,
            rng=rng,
        )

        self.assertEqual([token.sno for token in plan.inputs], [2, 1])
        self.assertEqual(plan.change, 5.00)
        self.assertEqual(rng.calls, 3)

    def test_duplicate_token_identity_and_successful_selection_purity(self):
        first_six = Token(6.00, 1)
        second_six = Token(6.00, 2)
        four = Token(4.00, 3)
        tokens = [first_six, second_six, four]
        original_tokens = list(tokens)

        plan = plan_randomized_adaptive_greedy_selection(
            tokens,
            10.00,
            probability=1.0,
            variant=RagVariant.Fit,
            rng=ConstantRng(0.0),
        )

        self.assertIs(plan.inputs[0], first_six)
        self.assertIs(plan.inputs[1], four)
        self.assertEqual(tokens, original_tokens)

    def test_insufficient_funds_is_pure(self):
        tokens = [Token(6.00, 1), Token(4.00, 2)]
        original_tokens = list(tokens)

        with self.assertRaises(InsufficientFundsError):
            plan_randomized_adaptive_greedy_selection(
                tokens, 10.01, probability=1.0, rng=ConstantRng(0.0)
            )

        self.assertEqual(tokens, original_tokens)

    def test_attempt_limit_counts_scans_then_uses_safety_top_up(self):
        rng = ConstantRng(0.5)
        with patch.object(RandomizedAdaptiveGreedyStrategy, "MAX_ATTEMPTS", 2):
            plan = plan_randomized_adaptive_greedy_selection(
                [Token(8.00, 1), Token(5.00, 2)],
                10.00,
                probability=0.0,
                variant=RagVariant.LargestFirst,
                rng=rng,
            )

        self.assertEqual(rng.calls, 4)
        self.assertEqual([token.sno for token in plan.inputs], [1, 2])
        self.assertEqual(plan.change, 3.00)


if __name__ == "__main__":
    unittest.main()
