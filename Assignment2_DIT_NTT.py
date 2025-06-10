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
                #Please write your code here
                t=montgomery(v,omega,prime)
                a[ind1]=(u+t)%prime
                a[ind2]=(u-t+prime)%
                count=count+1
            r=r+1
    return a

prime=3329
arr=[]
for i in range(256):
    arr.append((i)%prime)

arr1=radix2(arr,8,1,prime)
print((arr1))
            
