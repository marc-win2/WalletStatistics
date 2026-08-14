# Created by Marc Winstel on July 17, 2025
import numpy as np


MINIMUM_DENOMINATION = 0.01
DENOMINATION_DECIMAL_PLACES = 2


def roundToMinimumDenomination(value):
    """Round a monetary value to the smallest supported denomination."""
    return round(float(value), DENOMINATION_DECIMAL_PLACES)



class Token:
    """
    Class to represent a token with a value.
    """
    def __init__(self, value, serialno=None):
        roundedValue = roundToMinimumDenomination(value)
        if roundedValue < MINIMUM_DENOMINATION:
            raise ValueError(
                f"Token value must be at least {MINIMUM_DENOMINATION:.2f}."
            )
        self.value = roundedValue
        self.sno = serialno ## typical an integer in this simulation 

    def __repr__(self):
        return f"Token(v={self.value}, sno={self.sno})" 
    
    

class Wallet: 
    """
    Class to represent a wallet containing tokens.
    """
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = []
        self.tokens = tokens

    def addToken(self, token):
        """
        Add a token to the wallet.
        """
        self.tokens.append(token)

    def __repr__(self):
        return f"Wallet(tokens={self.tokens})"
    

    def searchTokenBySno(self, serialno):
        """
        Search for a token by its serial number.
        """
        for token in self.tokens:
            if token.sno == serialno:
                return token
        return None
    
    def selectTokenRandomly(self):
        """
        Select a token randomly from the wallet.
        """
        if not self.tokens:
            return None
        return np.random.choice(self.tokens)
    
    def removeTokenBySno(self, serialno):
        """
        Remove a token by its serial number.
        """
        self.tokens = [token for token in self.tokens if token.sno != serialno]
        return self.tokens
    
    def getTokenValue(self,serialno):
        """
        Get the value of a token by its serial number.
        """
        token = self.searchTokenBySno(serialno)
        if token is not None:
            return token.value
        else:
            return None 
        
    def getTokenCount(self):
        """
        Return the number of tokens in the wallet.
        """
        return len(self.tokens)
    
    def getTotalValue(self):
        """
        Get the total value of all tokens in the wallet.
        """
        return roundToMinimumDenomination(sum(token.value for token in self.tokens))

    def isEmpty(self):
        """
        Check if the wallet is empty.
        """
        return len(self.tokens) == 0
    
