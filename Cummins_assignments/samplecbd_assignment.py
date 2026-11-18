# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:27:25 2026

@author: suraj
"""


def bytes_to_bits(b):
    """
    Converts a byte array B of length m into an array of 8*m bits (FIPS 203 Algorithm 7).
    Least significant bit first.
    """
    l = len(b)
    a = bytearray(8 * l)
    for i in range(0, 8 * l, 8):
        x = b[i // 8]
        for j in range(8):
            a[i + j] = (x >> j) & 1
    return a


def sample_poly_cbd(eta, b, modq=3329):
    """
    SamplePolyCBD_eta (FIPS 203 Algorithm 8 / CRYSTALS-Kyber).
    Samples a degree-255 polynomial with coefficients from the Centered Binomial Distribution.
    """
    if isinstance(b, str):
        b = bytes.fromhex(b)

    b_bits = bytes_to_bits(b)
    
    f = [0] * 256
    for i in range(256):
        #write your code here
        
    return f


# Parameters for Kyber-512 / ML-KEM-512
eta = 2
modq = 3329
b = 'dd1620f671b666c3c6bca810ad17916b445d1c6ab0ffe85c2b317aa0897e363fc8af1634b114c11113a834a6a36a182655d79da985d69b646e7cf2f193db13369366255cc6fb1e1db922e0878d48a3abb664af0af3e5c77872ab292013d8c32cee205cfeaa6fb800320dc39bf13b2f5d2f4452a2ea8210a2218c9047fa5400a1'

res = sample_poly_cbd(eta, b, modq)
print("Length of polynomial:", len(res))
print("coefficients:", res)


