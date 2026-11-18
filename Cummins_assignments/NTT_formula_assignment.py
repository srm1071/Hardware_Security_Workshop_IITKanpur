# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:08:18 2026

@author: suraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:57:33 2026

@author: suraj
"""

def NTT(a, gamma, omega, n, q):    
    NTT_poly1= [] 
    #write your code here    
         
    return NTT_poly1


def bitreverse1(num):
    b1=format(num,'07b')
    binary=b1[::-1]
    decimal=int(binary,2)
    return decimal
a1=[]
a2=[]
for i in range(256):
    if i%2==0:
        a1.append(i)
    else:
        a2.append(i)
gamma = 1
omega = 17
n = 128
q = 3329

out_poly1 = NTT(a1, gamma, omega, n, q)
out_poly2 = NTT(a2, gamma, omega, n, q)
out=[]
for i in range(128):
    out.append(out_poly1[i])
    out.append(out_poly2[i])

print(out)
