# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 19:15:24 2023

@author: suraj
"""
import math

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

def modInverse(A, M):
    g = gcdExtended(A, M)
    if (g != 1):
        print()
    else:
        res = (x % M + M) % M
    return res


prime=2**23-2**13+1
omega=3073009
gamma=1753

poly1=[]
for i in range(256):
    poly1.append(i)   
ntt_poly1=[]
for i in range(256):
    ntt_poly1.append(0) 


n=256
sum1=0
for i in range(n):
    sum1=0
    for j in range(n):
      #Please enter your code here
	temp=pow(omega,i*j,prime)
        temp1=pow(gamma,j,prime)
        sum1=(sum1+poly1[j]*temp*temp1)%prime
    ntt_poly1[i]=sum1

print("ntt is: ",(ntt_poly1))
