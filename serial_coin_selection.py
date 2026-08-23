"""Serial, payment-level coin-selection strategies.

The existing simulator selects and removes one token at a time.  This module
instead plans a complete payment without touching a wallet, so strategies that
need to reason about an entire input set (such as Branch-and-Bound) can share a
single contract.  Applying a plan to a ``Wallet`` is deliberately left to the
caller.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import math
import random
from typing import Optional, Sequence, Tuple

from wallet import Token


_CENTS_PER_UNIT = Decimal("100")


def amount_to_cents(amount: float) -> int:
    """Convert a supported monetary amount to integer cents.

    ``Token`` values are already rounded to cents.  Converting through
    ``str`` avoids allowing binary floating-point representation to influence
    a selection decision.
    """
    decimal_amount = Decimal(str(amount))
    if not decimal_amount.is_finite():
        raise ValueError(f"Amount must be finite: {amount!r}")
    cents = decimal_amount * _CENTS_PER_UNIT
    rounded_cents = cents.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    if cents != rounded_cents:
        raise ValueError(f"Amount must be representable in cents: {amount!r}")
    return int(rounded_cents)


def cents_to_amount(cents: int) -> float:
    """Convert an integer-cent amount to the simulator's money representation."""
    return float(Decimal(cents) / _CENTS_PER_UNIT)


class InsufficientFundsError(ValueError):
    """Raised when a pure selection plan cannot cover its payment."""


@dataclass(frozen=True)
class SelectionPlan:
    """A complete, non-mutating plan for one payment.

    Monetary fields are exposed as the simulator's two-decimal floats, while
    all strategy arithmetic is performed in integer cents. ``changeless`` is
    retained as common strategy metadata, but the strategies implemented here
    all use ordinary-change semantics. ``bnb_fallback_used`` distinguishes a
    successful BnB search from its largest-first fallback.
    """

    inputs: Tuple[Token, ...]
    selected_total: float
    payment_amount: float
    change: float
    changeless: bool
    strategy: str
    bnb_fallback_used: bool = False

    @property
    def used_bnb_fallback(self) -> bool:
        """Compatibility-friendly spelling for BnB fallback metadata."""
        return self.bnb_fallback_used

    @property
    def selected_total_cents(self) -> int:
        return amount_to_cents(self.selected_total)

    @property
    def payment_amount_cents(self) -> int:
        return amount_to_cents(self.payment_amount)

    @property
    def change_cents(self) -> int:
        return amount_to_cents(self.change)


class CoinSelectionStrategy(ABC):
    """Contract for serial strategies that select a complete payment input set."""

    @abstractmethod
    def select(
        self,
        tokens: Sequence[Token],
        payment_amount: float,
        rng: Optional[object] = None,
    ) -> SelectionPlan:
        """Return a pure selection plan for a positive payment amount.

        Deterministic strategies ignore ``rng``.  It is part of the shared
        contract so randomized strategies can receive reproducible randomness
        at selection time.
        """


class GreedyStrategy(CoinSelectionStrategy):
    """Largest-fitting-first selection with smallest-token overshoot fallback."""

    name = "greedy"

    def select(
        self,
        tokens: Sequence[Token],
        payment_amount: float,
        rng: Optional[object] = None,
    ) -> SelectionPlan:
        del rng  # Greedy is deterministic; retain the shared strategy shape.
        payment_cents = amount_to_cents(payment_amount)
        if payment_cents <= 0:
            raise ValueError("Payment amount must be positive.")

        # Work against a list copy.  No caller-owned collection or Wallet is
        # mutated, and the selected entries remain the original Token objects.
        available = list(tokens)
        available_total_cents = sum(amount_to_cents(token.value) for token in available)
        if available_total_cents < payment_cents:
            raise InsufficientFundsError(
                "Insufficient wallet funds: payment requires "
                f"{cents_to_amount(payment_cents):.2f}, but only "
                f"{cents_to_amount(available_total_cents):.2f} is available."
            )

        selected = []
        selected_total_cents = 0
        while selected_total_cents < payment_cents:
            remaining_cents = payment_cents - selected_total_cents
            fitting = [
                token
                for token in available
                if amount_to_cents(token.value) <= remaining_cents
            ]
            # max/min return the first equal maximum/minimum, preserving the
            # caller's deterministic order for equal denominations.
            if fitting:
                chosen = max(fitting, key=lambda token: amount_to_cents(token.value))
            else:
                chosen = min(available, key=lambda token: amount_to_cents(token.value))

            chosen_index = next(
                index for index, token in enumerate(available) if token is chosen
            )
            available.pop(chosen_index)
            selected.append(chosen)
            selected_total_cents += amount_to_cents(chosen.value)

        change_cents = selected_total_cents - payment_cents
        return SelectionPlan(
            inputs=tuple(selected),
            selected_total=cents_to_amount(selected_total_cents),
            payment_amount=cents_to_amount(payment_cents),
            change=cents_to_amount(change_cents),
            changeless=False,
            strategy=self.name,
        )


def plan_greedy_selection(
    tokens: Sequence[Token], payment_amount: float, rng: Optional[object] = None
) -> SelectionPlan:
    """Convenience entry point for the Greedy payment-level strategy."""
    return GreedyStrategy().select(tokens, payment_amount, rng=rng)


class BranchAndBoundStrategy(CoinSelectionStrategy):
    """Inclusion-first subset search with a largest-first safety fallback.

    When no absolute overshoot is configured, the search accepts up to 20%
    of the current payment amount as change.
    """

    name = "branch_and_bound"
    fallback_name = "branch_and_bound_fallback"
    DEFAULT_MAX_OVERSHOOT_PERCENT = 20
    MAX_SEARCH_ATTEMPTS = 100_000
    MAX_UTXOS = 2_000

    def __init__(self, max_bnb_overshoot: Optional[float] = None):
        if max_bnb_overshoot is None:
            self.max_bnb_overshoot = None
            self._max_bnb_overshoot_cents = None
            return

        max_bnb_overshoot_cents = amount_to_cents(max_bnb_overshoot)
        if max_bnb_overshoot_cents < 0:
            raise ValueError("Maximum BnB overshoot must not be negative.")
        self.max_bnb_overshoot = cents_to_amount(max_bnb_overshoot_cents)
        self._max_bnb_overshoot_cents = max_bnb_overshoot_cents

    def select(
        self,
        tokens: Sequence[Token],
        payment_amount: float,
        rng: Optional[object] = None,
    ) -> SelectionPlan:
        del rng  # Branch-and-Bound is deterministic.
        payment_cents = amount_to_cents(payment_amount)
        if payment_cents <= 0:
            raise ValueError("Payment amount must be positive.")

        # Pairing each existing object with its value preserves object identity
        # while doing every comparison and total in integer cents.  sorted is
        # stable, so equal-valued tokens retain caller order for deterministic
        # inclusion-first ties.
        ordered = sorted(
            ((token, amount_to_cents(token.value)) for token in tokens),
            key=lambda item: item[1],
            reverse=True,
        )
        available_total_cents = sum(value_cents for _, value_cents in ordered)
        if available_total_cents < payment_cents:
            raise InsufficientFundsError(
                "Insufficient wallet funds: payment requires "
                f"{cents_to_amount(payment_cents):.2f}, but only "
                f"{cents_to_amount(available_total_cents):.2f} is available."
            )

        if len(ordered) > self.MAX_UTXOS:
            return self._fallback(ordered, payment_cents)

        if self._max_bnb_overshoot_cents is None:
            # Keep the default relative to each payment. Integer division
            # rounds down to cents so the limit never exceeds 20%.
            max_overshoot_cents = (
                payment_cents * self.DEFAULT_MAX_OVERSHOOT_PERCENT // 100
            )
        else:
            max_overshoot_cents = self._max_bnb_overshoot_cents
        upper_bound_cents = payment_cents + max_overshoot_cents
        suffix_sums = [0] * (len(ordered) + 1)
        for index in range(len(ordered) - 1, -1, -1):
            suffix_sums[index] = suffix_sums[index + 1] + ordered[index][1]

        # Stack entries contain the actual current subset, rather than a
        # reusable selection bitmap.
        stack = [(0, 0, ())]
        best_inputs = None
        best_total_cents = None
        attempts = 0

        while stack:
            # counter: once exhausted, even pending children are not visited.
            if attempts >= self.MAX_SEARCH_ATTEMPTS:
                break
            index, selected_total_cents, inputs = stack.pop()

            if selected_total_cents > upper_bound_cents:
                continue
            if selected_total_cents >= payment_cents:
                # The upper-bound test above makes this an accepted subset.
                if (
                    best_total_cents is None
                    or selected_total_cents < best_total_cents
                ):
                    best_total_cents = selected_total_cents
                    best_inputs = inputs
                if selected_total_cents == payment_cents:
                    break
                continue
            if index == len(ordered):
                continue
            if selected_total_cents + suffix_sums[index] < payment_cents:
                continue

            attempts += 1
            token, value_cents = ordered[index]
            # LIFO processing means push exclusion first to explore inclusion
            # first, matching the serial Rust traversal order.
            stack.append((index + 1, selected_total_cents, inputs))
            stack.append(
                (index + 1, selected_total_cents + value_cents, inputs + (token,))
            )

        if best_inputs is None:
            return self._fallback(ordered, payment_cents)

        # User-directed semantic override: accepted BnB overshoot creates
        # ordinary change, rather than treating excess as a changeless fee.
        return self._plan(
            inputs=best_inputs,
            selected_total_cents=best_total_cents,
            payment_cents=payment_cents,
            strategy=self.name,
            bnb_fallback_used=False,
        )

    def _fallback(self, ordered_tokens, payment_cents: int) -> SelectionPlan:
        """Select unconditional largest-first inputs until the payment is met."""
        selected = []
        selected_total_cents = 0
        for token, value_cents in ordered_tokens:
            if selected_total_cents >= payment_cents:
                break
            selected.append(token)
            selected_total_cents += value_cents

        # Balance was checked before entering BnB, so this indicates only an
        # internal invariant failure rather than a caller-visible shortfall.
        if selected_total_cents < payment_cents:
            raise RuntimeError("BnB fallback could not cover a funded payment.")

        return self._plan(
            inputs=tuple(selected),
            selected_total_cents=selected_total_cents,
            payment_cents=payment_cents,
            strategy=self.fallback_name,
            bnb_fallback_used=True,
        )

    @staticmethod
    def _plan(
        inputs: Tuple[Token, ...],
        selected_total_cents: int,
        payment_cents: int,
        strategy: str,
        bnb_fallback_used: bool,
    ) -> SelectionPlan:
        return SelectionPlan(
            inputs=inputs,
            selected_total=cents_to_amount(selected_total_cents),
            payment_amount=cents_to_amount(payment_cents),
            change=cents_to_amount(selected_total_cents - payment_cents),
            changeless=False,
            strategy=strategy,
            bnb_fallback_used=bnb_fallback_used,
        )


def plan_branch_and_bound_selection(
    tokens: Sequence[Token],
    payment_amount: float,
    max_bnb_overshoot: Optional[float] = None,
    rng: Optional[object] = None,
) -> SelectionPlan:
    """Convenience entry point for Branch-and-Bound payment selection."""
    return BranchAndBoundStrategy(max_bnb_overshoot).select(
        tokens, payment_amount, rng=rng
    )


class RagVariant(str, Enum):
    """Candidate ordering rules for Randomized Adaptive Greedy."""

    LargestFirst = "largest_first"
    Fit = "fit"
    SmallestFirstConsolidate = "smallest_first_consolidate"

    # Conventional Python spellings are aliases, while the Rust-style names
    # above keep the behavioral reference directly recognizable.
    LARGEST_FIRST = LargestFirst
    FIT = Fit
    SMALLEST_FIRST_CONSOLIDATE = SmallestFirstConsolidate


class RandomizedAdaptiveGreedyStrategy(CoinSelectionStrategy):
    """Serial Randomized Adaptive Greedy (RAG) payment selection."""

    MAX_ATTEMPTS = 10_000

    def __init__(
        self,
        probability: float,
        target_pool_size: Optional[int] = None,
        variant: RagVariant = RagVariant.Fit,
    ):
        if not isinstance(probability, (int, float)) or not math.isfinite(probability):
            raise ValueError("Probability must be finite and between 0 and 1.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")
        if target_pool_size is not None:
            if isinstance(target_pool_size, bool) or not isinstance(target_pool_size, int):
                raise ValueError("Target pool size must be a non-negative integer or None.")
            if target_pool_size < 0:
                raise ValueError("Target pool size must be a non-negative integer or None.")

        self.probability = float(probability)
        self.target_pool_size = target_pool_size
        self.variant = self._coerce_variant(variant)

    @staticmethod
    def _coerce_variant(variant) -> RagVariant:
        if isinstance(variant, RagVariant):
            return variant
        try:
            return RagVariant(variant)
        except ValueError as error:
            raise ValueError(f"Unsupported RAG variant: {variant!r}") from error

    def select(
        self,
        tokens: Sequence[Token],
        payment_amount: float,
        rng: Optional[object] = None,
    ) -> SelectionPlan:
        payment_cents = amount_to_cents(payment_amount)
        if payment_cents <= 0:
            raise ValueError("Payment amount must be positive.")

        # The caller's wallet/list remains untouched.  Selected entries are
        # always original Token instances from this copied working list.
        available = list(tokens)
        balance_cents = sum(amount_to_cents(token.value) for token in available)
        if balance_cents < payment_cents:
            raise InsufficientFundsError(
                "Insufficient wallet funds: payment requires "
                f"{cents_to_amount(payment_cents):.2f}, but only "
                f"{cents_to_amount(balance_cents):.2f} is available."
            )

        selection_target_cents = self._selection_target(
            payment_cents, balance_cents, len(available)
        )
        adaptive = selection_target_cents > payment_cents
        selected = []
        selected_total_cents = 0
        attempts = 0

        while (
            selected_total_cents < selection_target_cents
            and attempts < self.MAX_ATTEMPTS
        ):
            attempts += 1
            candidate = self._choose_candidate(
                available=available,
                adaptive=adaptive,
                remaining_cents=selection_target_cents - selected_total_cents,
                payment_cents=payment_cents,
                selected_total_cents=selected_total_cents,
                rng=rng,
            )
            if candidate is None:
                # ``None`` represents either Retry or Stop.  Retry is
                # distinguished by the loop condition; Stop only occurs when
                # no candidate remains or Fit has already covered payment.
                if not available or (
                    self.variant is RagVariant.Fit
                    and selected_total_cents >= payment_cents
                    and not self._has_fitting_denomination(
                        available,
                        selection_target_cents - selected_total_cents,
                    )
                ):
                    break
                continue

            self._remove_identity(available, candidate)
            selected.append(candidate)
            selected_total_cents += amount_to_cents(candidate.value)

        # Randomness/adaptive consumption must never leave an underfunded
        # payment.  This safety top-up is deliberately unconditional
        # largest-first and runs only while actual payment is uncovered.
        if selected_total_cents < payment_cents:
            for token in self._sorted_tokens(available, ascending=False):
                if selected_total_cents >= payment_cents:
                    break
                selected.append(token)
                selected_total_cents += amount_to_cents(token.value)

        if selected_total_cents < payment_cents:
            raise RuntimeError("RAG safety top-up could not cover a funded payment.")

        return SelectionPlan(
            inputs=tuple(selected),
            selected_total=cents_to_amount(selected_total_cents),
            payment_amount=cents_to_amount(payment_cents),
            change=cents_to_amount(selected_total_cents - payment_cents),
            changeless=False,
            strategy=self._strategy_name(),
        )

    def _selection_target(
        self, payment_cents: int, balance_cents: int, pool_size: int
    ) -> int:
        if self.target_pool_size is None or self.target_pool_size == 0:
            return payment_cents
        if pool_size <= self.target_pool_size:
            return payment_cents

        # Mirror Rust's two f64 operations followed by ``as u64`` truncation.
        # This is intentionally not algebraically reduced to integer division:
        # floating representation can make e.g. 7 * (61 / 7) truncate to 60.
        scale = float(pool_size) / float(self.target_pool_size)
        scaled_cents = int(float(payment_cents) * scale)
        return min(max(scaled_cents, payment_cents), balance_cents)

    def _choose_candidate(
        self,
        available: Sequence[Token],
        adaptive: bool,
        remaining_cents: int,
        payment_cents: int,
        selected_total_cents: int,
        rng: Optional[object],
    ) -> Optional[Token]:
        if self.variant is RagVariant.LargestFirst:
            return self._pick_unfiltered(available, False, rng)
        if self.variant is RagVariant.SmallestFirstConsolidate:
            return self._pick_unfiltered(available, adaptive, rng)
        return self._pick_fit(
            available,
            remaining_cents,
            payment_cents,
            selected_total_cents,
            rng,
        )

    def _pick_unfiltered(
        self, available: Sequence[Token], ascending: bool, rng: Optional[object]
    ) -> Optional[Token]:
        for _, denomination_tokens in self._denomination_snapshot(available, ascending):
            if self._random_draw(rng) < self.probability:
                return denomination_tokens[0]
        return None

    def _pick_fit(
        self,
        available: Sequence[Token],
        remaining_cents: int,
        payment_cents: int,
        selected_total_cents: int,
        rng: Optional[object],
    ) -> Optional[Token]:
        snapshot = self._denomination_snapshot(available, ascending=False)
        fitting_exists = False
        for value_cents, denomination_tokens in snapshot:
            if value_cents <= remaining_cents:
                fitting_exists = True
                if self._random_draw(rng) < self.probability:
                    return denomination_tokens[0]

        if fitting_exists:
            return None  # Retry the next scan.
        if selected_total_cents >= payment_cents:
            return None  # Stop; do not overshoot merely to reach the target.
        return snapshot[-1][1][0] if snapshot else None

    @staticmethod
    def _remove_identity(available, chosen: Token) -> None:
        chosen_index = next(
            index for index, token in enumerate(available) if token is chosen
        )
        available.pop(chosen_index)

    @staticmethod
    def _sorted_tokens(available: Sequence[Token], ascending: bool) -> Sequence[Token]:
        return sorted(
            available,
            key=lambda token: amount_to_cents(token.value),
            reverse=not ascending,
        )

    @staticmethod
    def _denomination_snapshot(available: Sequence[Token], ascending: bool):
        denominations = {}
        for token in available:
            value_cents = amount_to_cents(token.value)
            denominations.setdefault(value_cents, []).append(token)
        return sorted(denominations.items(), key=lambda item: item[0], reverse=not ascending)

    @staticmethod
    def _has_fitting_denomination(available: Sequence[Token], remaining_cents: int) -> bool:
        return any(
            amount_to_cents(token.value) <= remaining_cents for token in available
        )

    @staticmethod
    def _random_draw(rng: Optional[object]) -> float:
        if rng is None:
            return random.random()
        value = rng() if callable(rng) else rng.random()
        return float(value)

    def _strategy_name(self) -> str:
        return f"rag_{self.variant.value}"


def plan_randomized_adaptive_greedy_selection(
    tokens: Sequence[Token],
    payment_amount: float,
    probability: float,
    target_pool_size: Optional[int] = None,
    variant: RagVariant = RagVariant.Fit,
    rng: Optional[object] = None,
) -> SelectionPlan:
    """Convenience entry point for Randomized Adaptive Greedy selection."""
    return RandomizedAdaptiveGreedyStrategy(
        probability=probability,
        target_pool_size=target_pool_size,
        variant=variant,
    ).select(tokens, payment_amount, rng=rng)
