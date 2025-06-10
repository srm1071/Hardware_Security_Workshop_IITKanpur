# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:52:47 2024

@author: suraj
"""

n=3329
a=1057
b=1852
x, y = 0, 1

def find_a_bar(a,r):
    a_bar=(a<<r)%n
    return a_bar
    
def find_r(n):
    i=1
    r=1
    while(i):
        if(2**i<n):
            i=i+1
        else:
            r=2**i
            break
    return r,i

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

def montgomery(a,b,r,r_bar,n_bar,power):
   #write your code here

r,power=find_r(n)
a_bar=find_a_bar(a,power)
b_bar=find_a_bar(b,power)
r_bar=modInverse(r,n)
n_bar=int((r*r_bar)//n)
#print("r_bar and n_bar are",r,",",r_bar,",",n_bar,",",power)
result=montgomery(a,b,r,r_bar,n_bar,power)
print("final result is",result)
