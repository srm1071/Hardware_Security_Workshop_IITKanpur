# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:43:06 2026

@author: suraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:37:56 2026

@author: suraj
"""


import matplotlib.pyplot as plt

try:
    from Crypto.Hash import SHAKE128

    def get_xof(seed):
        return SHAKE128.new(seed)
except ImportError:
    import hashlib

    class XOFStream:
        """Stream wrapper around hashlib.shake_128 to provide a .read(n) method."""
        def __init__(self, seed):
            self.shake = hashlib.shake_128(seed)
            self.offset = 0
            self.buffer = b""

        def read(self, n):
            while len(self.buffer) < self.offset + n:
                self.buffer = self.shake.digest(len(self.buffer) + 672)
            data = self.buffer[self.offset : self.offset + n]
            self.offset += n
            return data

    def get_xof(seed):
        return XOFStream(seed)


class KyberSampler:
    def __init__(self, q=3329):
        self.q = q

    def sample_ntt(self, b):
        """
        Samples a polynomial of degree 255 in the NTT domain with coefficients
        uniformly distributed in [0, q-1] via rejection sampling.
        """
        if isinstance(b, str):
            b = bytes.fromhex(b)

        xof = get_xof(b)
        j = 0
        a = []

        while j < 256:
            # 1. Read 3 bytes (24 bits)
            c = xof.read(3)

            # 2. Unpack into two 12-bit candidate integers
            d1 = c[0] + 256 * (c[1] % 16)
            d2 = (c[1] // 16) + 16 * c[2]

            # 3. Rejection sampling: Write Your Code Here
            

        return a


sampler = KyberSampler(q=3329)

sample_seed = bytes.fromhex(
    '7c9935a0b07694aa0c6d10e4db6b1add2fd81a25ccb148032dcd739936737f2d'
)

res = sampler.sample_ntt(sample_seed)

print(f"Sampled {len(res)} coefficients in Z_{sampler.q}")
print("coefficients:", res)

