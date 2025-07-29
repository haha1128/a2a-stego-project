import random
import math

from .utils import lsb_bits2int, lsb_int2bits

# Uniform Steganography schemes

## Binary Uniform Steganography (Adapted from: Boris Ryabko and Daniil Ryabko. Information-theoretic approach to steganographic systems. In 2007 IEEE International Symposium on Information Theory, pages 2461–2464. IEEE, 2007)

### Time-step Encoding
def uni_binary_enc(bit_stream, n, **kwargs):
    if n==1:
        return 0,''
    
    def find_bin(n):
        powers = [i for i in range(n.bit_length()) if n & (1 << i)]

        r = random.randint(1, n)
        sum_powers = 0
        group = 0
        for i, power in enumerate(powers):
            sum_powers += 2**power
            if r <= sum_powers:
                group = i 
                break

        return sum_powers-2**powers[group], powers[group]

    
    temp, bits_num = find_bin(n)
    bits = bit_stream[:bits_num]

    idx =temp + lsb_bits2int([int(b) for b in bits])

    return idx,bits   #idx from 0 to n-1

### Time-step Decoding
def uni_binary_dec(idx, n, **kwargs):
    if n==1:
        return ''
    
    powers = [i for i in range(n.bit_length()) if n & (1 << i)]

    sum_powers = 0
    group = 0
    for i, power in enumerate(powers):
        sum_powers += 2**power
        if idx + 1 <= sum_powers:
            group = i  
            break    
    
    bits_num = powers[group]
    tmp = sum_powers-2**powers[group]

    bits = lsb_int2bits(idx - tmp, bits_num)
    bits = "".join([str(_) for _ in bits])
    return bits
    
    
    
    
## Cyclic_shift Uniform Steganography (by ours)    

### Time-step Encoding
def uni_cyclic_shift_enc(bit_stream, n, PRG, precision):
    if n==1:
        PRG.generate_random(n = precision)
        return 0,''
    
    ptr = PRG.generate_random(n = precision)
    R = math.floor(ptr * n) 

    k = math.floor(math.log2(n))
    t = n - 2**k
    bits = bit_stream[:k]
    bits_res = bit_stream[k] if k < len(bit_stream) else '0'    

    idx_sort = lsb_bits2int([int(b) for b in bits])
    if idx_sort < 2**k - t:
        return (idx_sort + R) % n, bits
    else:
        return (2 * (idx_sort - (2**k - t)) + (2**k - t) + R + int(bits_res)) % n, bits + bits_res

### Time-step Decoding
def uni_cyclic_shift_dec(idx, n, PRG, precision):
    if n==1:
        PRG.generate_random(n = precision)
        return ''
    
    ptr = PRG.generate_random(n = precision)
    R = math.floor(ptr * n) 

    k = math.floor(math.log2(n))
    t = n - 2**k
 
    idx_sort = (idx - R) % n

    if idx_sort < 2**k - t:
        bits = lsb_int2bits(idx_sort, k)
        bits = "".join([str(_) for _ in bits])
        return bits
    else:
        s1 = idx_sort - 2**k + t
        s_last = s1 % 2

        bits = lsb_int2bits((s1 - s_last)//2 + 2**k -t, k)
        bits = "".join([str(_) for _ in bits])
        if s_last == 0:
            return bits + '0'
        else:
            return bits + '1' 