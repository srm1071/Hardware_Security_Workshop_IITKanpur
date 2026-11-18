# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:11:09 2026

@author: suraj
"""

# -*- coding: utf-8 -*-
"""
CRYSTALS-Kyber (ML-KEM / FIPS 203): NTT, PWM, and INTT
"""

Q = 3329       # Kyber modulus
N = 256        # Degree
ZETA = 17      # Primitive 256-th root of unity (17^128 == -1 mod 3329)


def bit_reverse_7(k):
    """Reverses 7 bits of integer k."""
    res = 0
    for _ in range(7):
        res = (res << 1) | (k & 1)
        k >>= 1
    return res


# Precompute 128 twiddle factors in bit-reversed order: zetas[k] = zeta^(BitRev7(k)) mod q
ZETAS = [pow(ZETA, bit_reverse_7(k), Q) for k in range(128)]
INV_128 = pow(128, Q - 2, Q)  # 3303 (128^-1 mod 3329)


# =============================================================================
# 1. Forward NTT (FIPS 203 Algorithm 9)
# =============================================================================
def ntt(f):
    """
    Computes incomplete NTT of f in-place or returns transformed copy.
    f: list of 256 integers in Z_q.
    """
    r = list(f)
    i = 1
    length = 128
    while length >= 2:
        for start in range(0, 256, 2 * length):
            z = ZETAS[i]
            i += 1
            for j in range(start, start + length):
                t = (z * r[j + length]) % Q
                r[j + length] = (r[j] - t) % Q
                r[j] = (r[j] + t) % Q
        length //= 2
    return r


# =============================================================================
# 2. Point-Wise Multiplication / BaseMul (FIPS 203 Algorithm 10)
# =============================================================================
def base_case_multiply(a0, a1, b0, b1, gamma):
    #write your code here
    
    
    return c0, c1


def pwm(a_hat, b_hat):
    """
    Point-wise multiplication of two polynomials in NTT domain (MultiplyNTTs).
    """
    c_hat = [0] * 256
    for i in range(128):
        z = ZETAS[64 + i // 2]
        gamma = z if (i % 2 == 0) else (-z) % Q
        
        c0, c1 = base_case_multiply(
            a_hat[2 * i], a_hat[2 * i + 1],
            b_hat[2 * i], b_hat[2 * i + 1],
            gamma
        )
        c_hat[2 * i] = c0
        c_hat[2 * i + 1] = c1
    return c_hat


# =============================================================================
# 3. Inverse NTT (FIPS 203 Algorithm 11)
# =============================================================================
def intt(f_hat):
    """
    Computes Inverse NTT (NTT^-1) back to normal polynomial domain.
    """
    r = list(f_hat)
    i = 127
    length = 2
    while length <= 128:
        for start in range(0, 256, 2 * length):
            z = ZETAS[i]
            i -= 1
            for j in range(start, start + length):
                t = r[j]
                r[j] = (t + r[j + length]) % Q
                r[j + length] = (z * (r[j + length] - t)) % Q
        length *= 2

    # Scale by 128^-1 mod q
    for j in range(256):
        r[j] = (r[j] * INV_128) % Q
    return r


# =============================================================================
# 4. Reference Negacyclic Convolution (for Verification)
# =============================================================================
def schoolbook_negacyclic(a, b):
    """Direct polynomial multiplication modulo (X^256 + 1, Q)."""
    mult = [0] * (2 * N - 1)
    for i in range(N):
        for j in range(N):
            mult[i + j] = (mult[i + j] + a[i] * b[j]) % Q
    c = mult[:N]
    for k in range(N, 2 * N - 1):
        c[k - N] = (c[k - N] - mult[k]) % Q
    return c


# =============================================================================
# Verification Test
# =============================================================================
if __name__ == "__main__":
    import random
    random.seed(42)

    # Generate two random test polynomials in R_q
    a = [random.randint(0, Q - 1) for _ in range(N)]
    b = [random.randint(0, Q - 1) for _ in range(N)]

    # 1. Roundtrip test: INTT(NTT(a)) == a
    a_rec = intt(ntt(a))
    assert a_rec == a, "Roundtrip failed!"
    print("✓ Roundtrip Test Passed: INTT(NTT(a)) == a")

    # 2. Multiplication test: INTT(PWM(NTT(a), NTT(b))) == a * b mod (X^256 + 1)
    c_schoolbook = schoolbook_negacyclic(a, b)
    c_ntt = intt(pwm(ntt(a), ntt(b)))

    assert c_schoolbook == c_ntt, "Multiplication mismatch!"
    print("✓ Multiplication Test Passed: INTT(PWM(NTT(a), NTT(b))) == a * b mod (X^256 + 1)")
    print(f"Sample First 5 Output Coefficients: {c_ntt[:5]}")