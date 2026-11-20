# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:54:00 2026

@author: suraj
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import os

# ---------- Utilities ----------
def read_traces_csv(filename):
    traces = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            traces.append([float(x) for x in row])
    return np.array(traces)

def hw(x):
    return bin(x).count("1")

def montgomery1(a, b, r, r_bar, n_bar, power):
    n = 3329
    t = (a * b)
    td = t * n_bar
    m = (t * n_bar) % r
    inter_n = (m * n)
    k = t + (m * n)
    u = int((t + (m * n)) >> power)
    if u > n:
        return (u - n)
    else:
        return u

N_TRACES = 5000
KEY_SPACE = 3329
EXCLUDED_KEYS = {0, 767}

# ---------- Load polynomial values ----------
for coeffs in range(1):
    first_column = []
    with open('polynomials_reg0.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            first_column.append(int(row[0]))
    
    first_column = first_column[0:N_TRACES]
    
    H = np.zeros((N_TRACES, KEY_SPACE))
    for i, value in enumerate(first_column[:N_TRACES]):
        for k in range(KEY_SPACE):
            inter = montgomery1(value, k, 4096, 2704, 3327, 12)
            H[i, k] = hw(inter)
    
    P = read_traces_csv("kyber_power_traces_reg0.csv")
    P = P[:N_TRACES, :]
    
    valid_samples = np.std(P, axis=0) != 0
    P = P[:, valid_samples]
    
    N_SAMPLES = min(1000, P.shape[1])
    
    corrmat = np.zeros((KEY_SPACE, N_SAMPLES))
    
    for k1 in trange(KEY_SPACE, desc="CPA", smoothing=0.1):
        # Skip excluded keys
        if k1 in EXCLUDED_KEYS:
            continue
            
        h = H[:, k1]
        # Avoid division-by-zero if variance is zero
        if np.std(h) == 0:
            continue
    
        for k2 in range(N_SAMPLES):
            corr = np.corrcoef(h, P[:, k2])
            corrmat[k1, k2] = abs(corr[0, 1])   # extract Pearson coefficient
            
    corrmat1 = corrmat[:, :N_SAMPLES]
    
    # ---------- Key ranking (excluding specified keys) ----------
    for ex_k in EXCLUDED_KEYS:
        corrmat1[ex_k, :] = 0.0

    key_scores = np.nanmax(np.abs(corrmat1), axis=1)   # shape: (KEY_SPACE,)
    best_key = int(np.argmax(key_scores))
    best_corr = key_scores[best_key]
    
    # Absolute correlation for the top candidate
    abs_corr_key = np.abs(corrmat1[best_key, :])
    best_column = int(np.argmax(abs_corr_key))
    
    print("\n" + "="*40)
    print("Best key hypothesis:", best_key)
    print("Best leaking column:", best_column)
    print(f"Correlation at that point: {corrmat1[best_key, best_column]:.6f}")
    print("="*40)

    ranked_keys = np.argsort(key_scores)[::-1]
    TOPK = 10
    print("\nTop", TOPK, "key hypotheses (excluding keys 0 and 767):")
    for rank in range(TOPK):
        k = ranked_keys[rank]
        col = np.argmax(np.abs(corrmat1[k]))
        val = corrmat1[k, col]
        print(f"Rank {rank+1:2d} | Key={k:4d} | Col={col:5d} | Corr={val:.6f}")
        
    # ---------- Plot ----------
    plt.figure()
    plt.plot(corrmat1[best_key], linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("Correlation coefficient")
    plt.title(f"Correlation vs Sample Points (Key = {best_key})")
    plt.grid(True)
    plt.savefig("cpa_result" + str(coeffs) + ".jpg", dpi=200, bbox_inches="tight")
    plt.show()
