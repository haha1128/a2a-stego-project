import hashlib
import hmac
import torch

import numpy as np

# MSB
# e.g. [0, 1, 1, 1] looks like 0111=7
def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res

def msb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in strlist]

#lsb
# e.g. [0, 1, 1, 1] looks like 1110=14
def lsb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res

def lsb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def calculate_entropy(prob):
    mask = prob > 0
    prob_nonzero = prob[mask]
    log_prob_nonzero = torch.log2(prob_nonzero)
    entropy = -torch.sum(prob_nonzero * log_prob_nonzero)
    return entropy.item()

# This module's implementation is largely adapted from the following sources:
# Website: https://meteorfrom.space/
# Reference: G. Kaptchuk, T. M. Jois, M. Green, and A. D. Rubin, "Meteor: Cryptographically Secure Steganography for Realistic Distributions," in Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, Virtual Event Republic of Korea: ACM, Nov. 2021, pp. 1529–1548.
class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0
        self.test = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        self.test+=1
        xs = np.zeros(n, dtype=bool)
        for i in range(0,n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs
    
    def generate_random(self, n):
        xs = self.generate_bits(n)
        def binary_array_to_float(bin_array):
            decimal_value = 0
            for bit in bin_array:
                decimal_value = (decimal_value << 1) | bit
            
            max_value = (1 << len(bin_array)) - 1
            
            return decimal_value / max_value
        return binary_array_to_float(xs) 