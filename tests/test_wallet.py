import unittest
from unittest.mock import patch

from wallet import Token, Wallet, roundToMinimumDenomination


class WalletTests(unittest.TestCase):
    def test_values_are_rounded_to_cents(self):
        self.assertEqual(roundToMinimumDenomination(12.345), 12.35)
        self.assertEqual(Token(1.236).value, 1.24)

    def test_token_rejects_values_below_minimum_denomination(self):
        with self.assertRaises(ValueError):
            Token(0.004)

    def test_wallet_tracks_and_removes_tokens(self):
        wallet = Wallet([Token(1.25, 1), Token(2.50, 2)])

        self.assertEqual(wallet.getTokenCount(), 2)
        self.assertEqual(wallet.getTotalValue(), 3.75)
        self.assertEqual(wallet.getTokenValue(2), 2.50)

        wallet.removeTokenBySno(1)
        self.assertEqual(wallet.getTokenCount(), 1)
        self.assertIsNone(wallet.searchTokenBySno(1))

    def test_random_selection_returns_a_wallet_token(self):
        tokens = [Token(1.00, 1), Token(2.00, 2)]
        wallet = Wallet(tokens)
        with patch("wallet.np.random.choice", return_value=tokens[1]):
            self.assertIs(wallet.selectTokenRandomly(), tokens[1])

    def test_random_selection_uses_supplied_generator(self):
        tokens = [Token(1.00, 1), Token(2.00, 2)]
        wallet = Wallet(tokens)
        generator = unittest.mock.Mock()
        generator.choice.return_value = tokens[0]

        self.assertIs(wallet.selectTokenRandomly(rng=generator), tokens[0])
        generator.choice.assert_called_once_with(tokens)


if __name__ == "__main__":
    unittest.main()
