# Created by Marc Winstel on July 17, 2025
import numpy as np



class Token:
    """
    Class to represent a token with a value.
    """
    def __init__(self, value, serialno=None):
        self.value = value
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
    

