import numpy as np

N = 2

def testBit(int_type, offset):
    mask = 1 << offset
    return int_type & mask

len_a = 2**N
H = np.zeros((len_a, len_a))

for a in range(len_a):
    for i in range(N):
        j = np.mod(i + 1, N)
        
        if (testBit(a, i) > 0 and testBit(a, j) > 0) or (testBit(a, i) == 0 and testBit(a, j) == 0):
            H[a, a] += 0.25
        else:
            H[a, a] += - 0.25
            b = a ^ 2**(i+j)
            H[a, b] = 0.5

print([bin(a)[2:] for a in range(len_a)])
print(H)            
