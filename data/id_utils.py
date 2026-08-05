BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def to_base62(n):
    """
    Encodes a non-negative integer as a base62 string (0-9, A-Z, a-z).
    Keep identical to eqasim-switzerland's copy (data/utils.py).
    """
    n = int(n)
    if n == 0:
        return "0"

    digits = []
    while n > 0:
        n, remainder = divmod(n, 62)
        digits.append(BASE62_ALPHABET[remainder])

    return "".join(reversed(digits))
