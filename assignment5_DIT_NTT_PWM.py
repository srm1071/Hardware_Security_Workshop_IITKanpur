# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:07:22 2024

@author: suraj
"""

import numpy as np

def modInverse(A, M):
    g = gcdExtended(A, M)
    if (g != 1):
        print()
    else:
        res = (x % M + M) % M
    return res

def gcdExtended(a, b):
     global x, y
     if (a == 0):
         x = 0
         y = 1
         return b
     gcd = gcdExtended(b % a, a)
     x1 = x
     y1 = y
     x = y1 - (b // a) * x1
     y = x1
     return gcd

def montgomery(a,b, prime):
    r,power=4096,12
    r_bar=modInverse(r,prime)
    n_bar=int((r*r_bar)//prime)

    n=prime
    t=(a*b)%n
    td=t*n_bar
    m=(t*n_bar)%r
    inter_n=m*n
    k=t+(m*n)
    u=int((t+(m*n)) >> power)
    if(u>n):
        return (u-n)
    else:
        return u

def bitreverse4(a):
    b1=format(a,'07b')
    binary=b1[::-1]
    decimal=int(binary,2)
    return decimal

def bitreverse3(a):
    a1=[0]*(len(a))
    a2=a
    dec=[]
    for i in range(len(a)):
        b1=format(i,'07b')
        binary=b1[::-1]
        decimal=int(binary,2)
        dec.append(decimal)
        a1[decimal]=a2[i]
    return a1

def div2_inv(t1d,u1d,gamma1,prime):
    t1=int((t1d+u1d)%prime)
    u1=int((t1d-u1d+prime)%prime)
    if(t1%2!=0):
        a1=(t1+prime)>>1;
    else:
        a1=t1>>1;

    b1=montgomery(u1,gamma1, prime)
    return a1,b1

def radix2(a,k1,k2,prime):
    count=0
    omegaMain=17
    omega_inv=1175
    N=256

    use_r=4096%prime
    omegas=[]
    for i in range(128):
        omegas.append(i)
    omegas=bitreverse3(omegas)
    for i in range(128):
        omegas[i]=(omegaMain**omegas[i]*use_r)%prime
    r=1
    vcheck=0
    flagarr=[]
    for i in range(k1,k2,-1):
        m=2**i
        #r=0
        for k in range(0,256,m):
            mrev=bitreverse4(m//2)
            omega=omegas[r]
            for j in range(0,m//2):
                ind1=j+k
                ind2=j+k+(m//2)
                u=a[ind1]
                v=a[ind2]
                t=montgomery(v,omega,prime)
                a[ind1]=(u+t)%prime
                a[ind2]=(u-t+prime)%prime
                count=count+1
            r=r+1
    return a

def radix2_inverse(a,prime):
    count=0
    omegaMain=17
    omega_inv=1175
    N=256

    use_r=4096%prime
    inv2=modInverse(2, prime)
    omegas=[]
    for i in range(1,128):
        temp3=(omega_inv**i)%prime
        temp3=(temp3*use_r*inv2)%prime
        omegas.append(temp3)
    omegas=bitreverse3(omegas)

    count1=0
    t=2
    m=128
    while(m>1):
        m=m//2
        j1=0
        h=m
        for i in range(h):
            j2=j1+t-1
            s=omegas[count1]
            count1=count1+1
            for j in range(j1,j2+1):
                u=a[j]
                v=a[j+t]
                k1,k2=div2_inv(u,v,s,prime)
                a[j]=k1
                a[j+t]=k2
                count=count+1
            j1=j1+(2*t)
        t=2*t
    return a

def base_multiplier(a0, a1, b0, b1, zeta):
    r = [ 0 for x in range(0,2)]
    r[0] = montgomery(a1, b1,3329)
    r[0] = montgomery(r[0], zeta,3329)
    r[0] = (r[0] + montgomery(a0,b0,3329))%3329
    r[1] = montgomery(a0, b1,3329)
    r[1] = (r[1] + montgomery(a1, b0,3329))%3329
    return r

def pointwise(poly_a, poly_b):
    NTT_ZETAS=[767, 1201, 1434, 2610, 2382, 505, 226, 1261, 2278, 455, 1555, 2092, 2973, 341, 324, 924, 660, 2622, 1681, 232, 2653, 3004, 316, 408, 1830, 1520, 878, 38, 1152, 1066, 2692, 526, 1949, 873, 1630, 1936, 2624, 2798, 2063, 1568, 2529, 1664, 3309, 2039, 315, 2008, 424, 716, 987, 2075, 3104, 468, 1047, 2616, 1441, 1397, 2888, 3181, 738, 995, 28, 1806, 2331, 2209, 3052, 443, 1075, 1093, 546, 1927, 513, 1463, 2107, 1077, 3132, 2274, 606, 2468, 2179, 2392, 1233, 1297, 1945, 615, 1824, 1133, 2043, 278, 1149, 2537, 1610, 646, 2939, 1477, 2487, 2284, 3172, 1525, 1078, 2951, 1331, 960, 1781, 24, 3045, 1656, 2989, 1373, 2026, 846, 550, 2185, 134, 1985, 2833, 1298, 1154, 1195, 1194, 446, 2490, 813, 2559, 270, 476, 741, 3008, 934]

    for i in range(0, 256 // 4):
        rx = base_multiplier(poly_a[4*i+0], poly_a[4*i+1],
                             poly_b[4*i+0], poly_b[4*i+1],
                             (NTT_ZETAS[64 + i]))
        ry = base_multiplier(poly_a[4*i+2], poly_a[4*i+3],
                             poly_b[4*i+2], poly_b[4*i+3],
                             (-1 * NTT_ZETAS[64 + i]))
        poly_a[4 * i + 0] = rx[0]
        poly_a[4 * i + 1] = rx[1]
        poly_a[4 * i + 2] = ry[0]
        poly_a[4 * i + 3] = ry[1]
    return poly_a



prime=3329
arr=[]
for i in range(256):
    arr.append((i)%prime)

arr1=radix2(arr,8,1,prime)

arr1=pointwise(arr1,arr1)

arr2=radix2_inverse(arr1,prime)

for i in range(len(arr2)):
    arr2[i]=montgomery(arr2[i],(4096**2)%3329,prime)

print(arr1)
