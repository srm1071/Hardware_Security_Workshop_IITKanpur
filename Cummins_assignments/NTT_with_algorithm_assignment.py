# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:19:59 2026

@author: suraj
"""

def radix2_kyber(x):
    r=1
    for i in range(8,1,-1):
        print('.............................', i)
        m=2**i
        print("r:", r)
        for k in range(0,n,m):
            omega=psi[r]
            for j in range(m//2):
                #write your code here
                #indices are (j+k) and (j+k+m//2)
                
            r=r+1
            
    return x
q=3329
I_psi=[]
for i in range(128):
    I_psi.append(pow(17, i,q))

def bitreverse2(num):
    b1=format(num,'07b')
    binary=b1[::-1]
    decimal=int(binary,2)
    return decimal

def bitreverse1(a):
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
psi=bitreverse1(I_psi)
n=256
x=[]
for i in range(n):
    x.append(i)
out=radix2_kyber(x)
print(out)